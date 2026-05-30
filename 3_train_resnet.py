"""
3_train_resnet.py
==================
ResNet18 六通道输入 — 二分类(涨/跌) + 混合精度 + EarlyStopping + 强正则化。

改进:
  - 数据增强: RandomErasing + GaussianNoise
  - 强正则化: Dropout(0.5) + weight_decay=1e-3 + Label Smoothing(0.02)
  - 二分类: 仅预测涨/跌，丢弃震荡样本（阈值±7%）
  - 第一层 kaiming_normal_ 初始化 (更合理的 6ch 权重)
"""

import os
import gc
import logging
from contextlib import nullcontext
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast
from torchvision.models import resnet18

from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, accuracy_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
SAMPLE_DIR     = BASE_DIR / "samples"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR        = BASE_DIR / "logs"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── 超参数 ────────────────────────────────────────────
BATCH_SIZE      = 64
NUM_EPOCHS      = 80
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-3
LABEL_SMOOTHING = 0.02
PATIENCE        = 15
NUM_WORKERS     = 0   # Windows spawn 兼容，设为 0 避免 pickle 错误
RANDOM_SEED     = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

CLASS_NAMES = ["下跌", "上涨"]
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

log.info("设备: %s | 混合精度: %s | LabelSmoothing: %.2f",
         DEVICE, USE_AMP, LABEL_SMOOTHING)


# ═══════════════════════════════════════════════════════
#  数据增强
# ═══════════════════════════════════════════════════════

class GaussianNoise:
    """添加高斯噪声（仅在训练时）。"""
    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomErasing3D:
    """对 (C, H, W) 张量的随机擦除 (Cutout 变体)。"""
    def __init__(self, p: float = 0.3, scale: tuple = (0.02, 0.1)):
        self.p = p
        self.scale = scale

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return tensor
        c, h, w = tensor.shape
        area = h * w
        er_area = area * (self.scale[0] + torch.rand(1).item() *
                          (self.scale[1] - self.scale[0]))
        aspect = torch.rand(1).item() * 2 + 0.5
        er_h = int(np.sqrt(er_area * aspect))
        er_w = int(np.sqrt(er_area / aspect))
        er_h = min(er_h, h - 1)
        er_w = min(er_w, w - 1)
        if er_h <= 0 or er_w <= 0:
            return tensor
        top = torch.randint(0, h - er_h + 1, (1,)).item()
        left = torch.randint(0, w - er_w + 1, (1,)).item()
        tensor[:, top:top + er_h, left:left + er_w] = torch.rand(1).item()
        return tensor


train_aug = [
    GaussianNoise(std=0.02),
    RandomErasing3D(p=0.3),
]


# ═══════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════

class KLineDataset(Dataset):
    """加载 6 通道 K 线图样本。"""

    def __init__(self, npz_paths: list[Path], augment: bool = False):
        self.augment = augment
        self.aug_list = train_aug if augment else []

        images_list, labels_list = [], []
        self.stocks, self.dates = [], []

        for npz_path in npz_paths:
            data = np.load(npz_path)
            n = len(data["labels"])
            images_list.append(data["images"])
            labels_list.append(data["labels"])
            stock = str(data["ts_code"]) if "ts_code" in data else npz_path.stem
            self.stocks.extend([stock] * n)
            self.dates.extend(list(data.get("dates", [""] * n)))

        self.images = np.concatenate(images_list, axis=0).astype(np.float32) / 255.0
        self.labels = np.concatenate(labels_list, axis=0).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]                         # (H, W, C)
        img = np.transpose(img, (2, 0, 1)).copy()      # (C, H, W)
        tensor = torch.as_tensor(img, dtype=torch.float32)

        for aug in self.aug_list:
            tensor = aug(tensor)

        return tensor, self.labels[idx]


# ═══════════════════════════════════════════════════════
#  SmallCNN (6ch → 2 classes, ~61K 参数)
# ═══════════════════════════════════════════════════════

class SmallCNN(nn.Module):
    """轻量级4层CNN — 6通道128×128输入, 2分类输出, ~61K参数。"""
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 128→64, 6→16
            nn.Conv2d(6, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64→32, 16→32
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 32→16, 32→64
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 16→1, 64→64 (全局池化)
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))


# ═══════════════════════════════════════════════════════
#  EarlyStopping
# ═══════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 15, mode: str = "max",
                 min_delta: float = 0.0005):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        improved = (score > self.best_score + self.min_delta
                    if self.mode == "max"
                    else score < self.best_score - self.min_delta)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                log.info("EarlyStopping: %d 轮未提升", self.patience)
        return self.should_stop


# ═══════════════════════════════════════════════════════
#  SAM (Sharpness-Aware Minimization)
# ═══════════════════════════════════════════════════════

class SAM:
    """SAM 优化器包装器 — 两步梯度更新, 引导权重进入平坦区域。"""
    def __init__(self, base_optimizer, rho: float = 0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.e_w = {}  # 扰动缓存

    def _grad_norm(self) -> torch.Tensor:
        norms = []
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.data.norm(2))
        return torch.norm(torch.stack(norms)) if norms else torch.tensor(0.0)

    @torch.no_grad()
    def first_step(self):
        """沿梯度方向扰动权重。"""
        grad_norm = self._grad_norm()
        if grad_norm == 0:
            return
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                e = p.grad * scale
                self.e_w[p] = e
                p.add_(e)

    @torch.no_grad()
    def second_step(self):
        """恢复权重并执行优化器更新。"""
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p in self.e_w:
                    p.sub_(self.e_w[p])
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()
        self.e_w.clear()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def __getattr__(self, name):
        return getattr(self.base_optimizer, name)


# ═══════════════════════════════════════════════════════
#  训练 / 验证
# ═══════════════════════════════════════════════════════

def train_one_epoch(model, loader, sam_optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # ── SAM first step: 扰动权重 ──────────────
        sam_optimizer.zero_grad()
        with autocast('cuda') if USE_AMP else nullcontext():
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        sam_optimizer.first_step()

        # ── SAM second step: 更新权重 ──────────────
        sam_optimizer.base_optimizer.zero_grad()
        with autocast('cuda') if USE_AMP else nullcontext():
            logits = model(images)
            loss = criterion(logits, labels)
        loss.backward()
        sam_optimizer.second_step()

        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return total_loss / n, acc, f1


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if USE_AMP:
            with autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro")
    return total_loss / n, acc, f1, np.array(all_preds), np.array(all_labels)


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(RANDOM_SEED)

    npz_files = sorted(SAMPLE_DIR.glob("*.npz"))
    if not npz_files:
        log.error("未找到 .npz 样本，请先运行 2_build_images.py")
        return

    # ── 股票级别划分 ───────────────────────────────
    stock_list = sorted(set(f.stem for f in npz_files))
    n_stocks = len(stock_list)
    log.info("%d 只股票, %d 个 npz 文件", n_stocks, len(npz_files))

    indices = np.random.permutation(n_stocks)
    n_train = int(n_stocks * TRAIN_RATIO)
    n_val   = int(n_stocks * VAL_RATIO)

    train_stocks = {stock_list[i] for i in indices[:n_train]}
    val_stocks   = {stock_list[i] for i in indices[n_train:n_train + n_val]}
    test_stocks  = {stock_list[i] for i in indices[n_train + n_val:]}

    log.info("训练:%d  验证:%d  测试:%d 只股票",
             len(train_stocks), len(val_stocks), len(test_stocks))

    train_files = [f for f in npz_files if f.stem in train_stocks]
    val_files   = [f for f in npz_files if f.stem in val_stocks]
    test_files  = [f for f in npz_files if f.stem in test_stocks]

    # ── Dataset ────────────────────────────────────
    train_ds = KLineDataset(train_files, augment=True)
    val_ds   = KLineDataset(val_files, augment=False)
    test_ds  = KLineDataset(test_files, augment=False)

    log.info("样本: 训练 %d, 验证 %d, 测试 %d",
             len(train_ds), len(val_ds), len(test_ds))

    # 类别权重
    label_counts = Counter(train_ds.labels.tolist())
    log.info("训练集分布: %s", dict(sorted(label_counts.items())))
    cls_weights = 1.0 / np.bincount(train_ds.labels, minlength=NUM_CLASSES)
    sample_weights = cls_weights[train_ds.labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    # ── 模型 ───────────────────────────────────────
    model = SmallCNN().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log.info("SmallCNN 参数量: %d", total_params)

    # Label Smoothing
    cls_tensor = torch.tensor(cls_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=cls_tensor,
        label_smoothing=LABEL_SMOOTHING,
    )

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    sam_optimizer = SAM(optimizer, rho=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6,
    )
    early_stop = EarlyStopping(patience=PATIENCE, mode="max")

    # ── 训练 ───────────────────────────────────────
    best_val_f1 = 0.0
    best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    history = {"train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, sam_optimizer, criterion)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion)

        scheduler.step(val_f1)

        for k, v in zip(history.keys(),
                        [train_loss, train_acc, train_f1,
                         val_loss, val_acc, val_f1]):
            history[k].append(v)

        lr_now = optimizer.param_groups[0]["lr"]
        log.info("Ep %2d | lr %.2e | "
                 "T_loss:%.4f T_acc:%.3f T_f1:%.4f | "
                 "V_loss:%.4f V_acc:%.3f V_f1:%.4f",
                 epoch, lr_now,
                 train_loss, train_acc, train_f1,
                 val_loss, val_acc, val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
            }, best_ckpt_path)
            log.info("  >> 保存最佳模型 (F1=%.4f)", val_f1)

        if early_stop(val_f1, epoch):
            break

    log.info("训练完成。最佳 Val F1: %.4f (epoch %d)",
             best_val_f1, early_stop.best_epoch)

    # -- 保存历史 --
    pd.DataFrame(history).to_csv(LOG_DIR / "training_history.csv", index=False)

    # -- 测试集 --
    log.info("\n测试集评估...")
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_acc, test_f1, test_preds, test_labels = validate(
        model, test_loader, criterion)

    log.info("Test Loss: %.4f  Acc: %.4f  Macro-F1: %.4f",
             test_loss, test_acc, test_f1)

    report = classification_report(test_labels, test_preds,
                                   target_names=CLASS_NAMES, digits=4)
    log.info("\n%s", report)
    cm = confusion_matrix(test_labels, test_preds)
    log.info("混淆矩阵:\n%s", cm)

    # 保存
    np.savez(LOG_DIR / "test_results.npz",
             preds=test_preds, labels=test_labels, confusion_matrix=cm)
    with open(LOG_DIR / "test_report.txt", "w") as f:
        f.write("Test Acc: {:.4f}  Macro-F1: {:.4f}  "
                "BestVal-F1: {:.4f}\n\n".format(test_acc, test_f1, best_val_f1))
        f.write(report)
        f.write("\n混淆矩阵:\n{}\n".format(cm))

    log.info("结果已保存到 %s", LOG_DIR)

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
