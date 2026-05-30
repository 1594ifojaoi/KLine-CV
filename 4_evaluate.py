"""
4_evaluate.py
==============
完整评估: 混淆矩阵 + 真实回测 + 金融指标。

改进:
  - 真实回测: 使用原始价格计算持有期收益（非 ±1% 虚构）
  - 概率阈值: P>threshold 才交易，否则观望
  - 金融指标: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate, Profit Factor
  - 按股票 + 全局分别汇报
"""

import os
import logging
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torchvision.models import resnet18

from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
SAMPLE_DIR      = BASE_DIR / "samples"
DATA_DIR        = BASE_DIR / "data"
CHECKPOINT_DIR  = BASE_DIR / "checkpoints"
LOG_DIR         = BASE_DIR / "logs"
EVAL_DIR        = BASE_DIR / "eval_results"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ── 参数 ──────────────────────────────────────────────
BATCH_SIZE        = 64
NUM_WORKERS       = 0   # Windows spawn 兼容
RANDOM_SEED       = 42
PROB_THRESHOLD    = 0.6           # 概率阈值: P>0.6 才交易
HOLDING_DAYS      = 5             # 持仓天数（与标签预测窗口一致）
RISK_FREE_RATE    = 0.03          # 年化无风险利率 3%

CLASS_NAMES = ["下跌", "上涨"]
NUM_CLASSES = len(CLASS_NAMES)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


# ═══════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════

class KLineDataset(Dataset):
    def __init__(self, npz_paths: list[Path]):
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

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = np.transpose(self.images[idx], (2, 0, 1)).copy()
        return torch.as_tensor(img, dtype=torch.float32), self.labels[idx]


# ═══════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════

class SmallCNN(nn.Module):
    """轻量级4层CNN — 架构与训练脚本一致。"""
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ═══════════════════════════════════════════════════════
#  金融指标
# ═══════════════════════════════════════════════════════

def calc_financial_metrics(returns: np.ndarray,
                           benchmark_returns: np.ndarray | None = None,
                           rf_annual: float = RISK_FREE_RATE) -> dict:
    """
    计算核心量化指标。
    returns: 策略日收益率序列
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)

    if n < 2:
        return {"error": "数据不足"}

    # 累积收益
    cumulative = np.cumprod(1.0 + returns)
    total_return = cumulative[-1] - 1.0

    # 年化收益率 (假设252个交易日)
    ann_return = (1.0 + total_return) ** (252.0 / n) - 1.0

    # 年化波动率
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe Ratio
    sharpe = (ann_return - rf_annual) / ann_vol if ann_vol > 0 else 0.0

    # Max Drawdown
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()

    # Calmar Ratio
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

    # Sortino Ratio (下行波动率)
    downside = returns[returns < 0]
    down_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0.0
    sortino = (ann_return - rf_annual) / down_vol if down_vol > 0 else 0.0

    # Win Rate
    wins = (returns > 0).sum()
    total_trades = n
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Profit Factor
    gross_profit = returns[returns > 0].sum() if wins > 0 else 0.0
    gross_loss = abs(returns[returns < 0].sum()) if (total_trades - wins) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # 超额收益 vs 基准
    alpha = 0.0
    beta = 0.0
    if benchmark_returns is not None and len(benchmark_returns) == n:
        bench_arr = np.asarray(benchmark_returns, dtype=np.float64)
        excess_ret = returns - bench_arr
        ann_excess = (1.0 + excess_ret.mean()) ** 252 - 1.0
        alpha = ann_excess
        cov = np.cov(returns, bench_arr)
        if cov.size == 4:
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": total_trades,
        "n_wins": int(wins),
        "alpha": alpha,
        "beta": beta,
    }


# ═══════════════════════════════════════════════════════
#  真实回测
# ═══════════════════════════════════════════════════════

def load_price_data() -> dict[str, pd.DataFrame]:
    """加载所有股票的前复权价格数据。"""
    price_map = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        if csv_path.stem == "manifest":
            continue
        try:
            df = pd.read_csv(csv_path, parse_dates=["trade_date"])
            stock = csv_path.stem.replace("_", ".")
            price_map[stock] = df.set_index("trade_date")
        except Exception:
            pass
    return price_map


def run_real_backtest(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    stocks: list[str],
    dates: list[str],
    price_map: dict[str, pd.DataFrame],
    prob_threshold: float = PROB_THRESHOLD,
    holding_days: int = HOLDING_DAYS,
) -> dict:
    """
    真实回测 (时间序列):
      - 当 P(上涨) > threshold → 做多，持有 holding_days 个交易日
      - 每只股票同一时间最多一个仓位（避免重复计数）
      - 按真实时间轴构建每日权益曲线
      - 多只股票同时持仓时等权平均日收益
    """
    log.info("运行真实回测 (threshold=%.2f, holding=%d天)...",
             prob_threshold, holding_days)

    # ── 1. 每只股票的日收益率 ──────────────────────────
    stock_daily_ret = {}
    for stock, pdf in price_map.items():
        close_col = "close_adj" if "close_adj" in pdf.columns else "close"
        if close_col not in pdf.columns:
            continue
        ret = pdf[close_col].pct_change()
        ret = ret[ret.notna()]  # 去掉 NaN（第一天）
        if len(ret) > 0:
            stock_daily_ret[stock] = ret

    # ── 2. 组装信号 DataFrame ──────────────────────────
    df = pd.DataFrame({
        "stock": stocks,
        "date": pd.to_datetime(dates),
        "p_up": probabilities[:, 1],
    })
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)

    # ── 3. 逐只股票生成持仓日期 ──────────────────────
    stock_in_position = defaultdict(set)  # stock → {date1, date2, ...}
    trade_log = []

    for stock, grp in df.groupby("stock"):
        if stock not in price_map or stock not in stock_daily_ret:
            continue

        price_df = price_map[stock]
        close_col = "close_adj" if "close_adj" in price_df.columns else "close"
        prices = price_df[close_col]
        grp = grp.sort_values("date").reset_index(drop=True)

        for _, row in grp.iterrows():
            sig_date = row["date"]
            p_up = row["p_up"]

            if p_up <= prob_threshold:
                continue

            # 信号日之后的交易日
            future = prices[prices.index > sig_date]
            if len(future) < holding_days:
                continue

            hold_dates = list(future.index[:holding_days])
            entry_date = hold_dates[0]
            exit_date  = hold_dates[-1]

            # 避免同只股票重叠持仓
            if any(d in stock_in_position[stock] for d in hold_dates):
                continue

            entry_price = prices.loc[entry_date]
            exit_price  = prices.loc[exit_date]
            hold_ret = (exit_price - entry_price) / entry_price

            for d in hold_dates:
                stock_in_position[stock].add(d)

            trade_log.append({
                "stock": stock, "signal_date": sig_date,
                "entry_date": entry_date, "exit_date": exit_date,
                "return": hold_ret, "p_up": p_up,
            })

    trade_df = pd.DataFrame(trade_log)
    if len(trade_df) == 0:
        log.warning("无有效交易信号")
        return {"daily_returns": pd.Series(dtype=float),
                "benchmark_daily": pd.Series(dtype=float),
                "metrics": {"n_trades": 0}, "trade_log": trade_df}

    # ── 4. 构建每日组合收益序列 ───────────────────────
    daily_ret_map = defaultdict(list)  # date → [ret1, ret2, ...]

    for stock, pos_dates in stock_in_position.items():
        if stock not in stock_daily_ret:
            continue
        sret = stock_daily_ret[stock]
        for d in pos_dates:
            if d in sret.index:
                daily_ret_map[d].append(sret.loc[d])

    # 组合日收益 = 当天所有持仓股票日收益的等权平均
    portfolio_daily = pd.Series({
        d: np.mean(rets) for d, rets in daily_ret_map.items()
    }).sort_index()

    # ── 5. 基准: 全部股票等权日收益 ────────────────────
    all_stock_rets = []
    for stock, sret in stock_daily_ret.items():
        all_stock_rets.append(sret)
    if all_stock_rets:
        benchmark_daily = pd.concat(all_stock_rets, axis=1).mean(axis=1)
        # 只取有组合收益的日期区间
        if len(portfolio_daily) > 0:
            start, end = portfolio_daily.index[0], portfolio_daily.index[-1]
            benchmark_daily = benchmark_daily.loc[start:end]
    else:
        benchmark_daily = pd.Series(dtype=float)

    # ── 6. 计算金融指标 ──────────────────────────────
    metrics = calc_financial_metrics(
        portfolio_daily.values,
        benchmark_daily.values if len(benchmark_daily) == len(portfolio_daily) else None,
    )
    metrics["n_trades"] = len(trade_df)
    metrics["n_days"] = len(portfolio_daily)

    cum_ret = (1.0 + portfolio_daily).cumprod().values
    bench_cum = (1.0 + benchmark_daily).cumprod().values if len(benchmark_daily) > 0 else np.ones(1)

    return {
        "daily_returns": portfolio_daily,
        "cumulative_return": cum_ret,
        "benchmark_return": bench_cum,
        "metrics": metrics,
        "trade_log": trade_df,
    }


# ═══════════════════════════════════════════════════════
#  OpenCV 可视化
# ═══════════════════════════════════════════════════════

def draw_confusion_matrix(cm, class_names, save_path):
    n = len(class_names)
    cs = 100
    m = 70
    iw = m * 2 + n * cs
    ih = m * 2 + n * cs + 50
    img = np.ones((ih, iw, 3), dtype=np.uint8) * 248
    cm_n = cm.astype(np.float32) / (cm.max() or 1)

    for i in range(n):
        for j in range(n):
            x1, y1 = m + j * cs, m + i * cs
            intense = int(cm_n[i, j] * 220)
            color = ((200 - intense, 230, 200 - intense) if i == j
                     else (200 - intense, 200 - intense, 230))
            cv2.rectangle(img, (x1, y1), (x1 + cs, y1 + cs), color, -1)
            cv2.rectangle(img, (x1, y1), (x1 + cs, y1 + cs), (180, 180, 180), 1)
            text = str(cm[i, j])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(img, text,
                        (x1 + (cs - tw) // 2, y1 + (cs + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    for i, name in enumerate(class_names):
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(img, name, (m - tw - 12, m + i * cs + cs // 2 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
        cv2.putText(img, name, (m + j * cs + cs // 2 - tw // 2, m + n * cs + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.putText(img, "Actual", (10, m + n * cs // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.imwrite(str(save_path), img)
    log.info("混淆矩阵: %s", save_path)


def draw_backtest_curve(cum_ret, bench_ret, save_path):
    w, h = 1000, 560
    ml, mr, mt, mb = 90, 30, 50, 70
    pw, ph = w - ml - mr, h - mt - mb
    img = np.ones((h, w, 3), dtype=np.uint8) * 248

    cv2.line(img, (ml, mt), (ml, h - mb), (50, 50, 50), 1)
    cv2.line(img, (ml, h - mb), (w - mr, h - mb), (50, 50, 50), 1)

    n = len(cum_ret)
    if n < 2:
        cv2.imwrite(str(save_path), img)
        return

    flat = np.concatenate([cum_ret, bench_ret]) if len(bench_ret) == n else cum_ret
    vmin, vmax = flat.min(), flat.max()
    rng = vmax - vmin or 1.0

    def _pts(arr):
        xv = np.linspace(ml, w - mr, len(arr))
        yv = mt + (vmax - arr) / rng * ph
        return np.column_stack([xv, yv]).astype(np.int32)

    # 1.0 参考线
    one_y = int(np.clip(mt + (vmax - 1.0) / rng * ph, mt, h - mb))
    cv2.line(img, (ml, one_y), (w - mr, one_y), (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(img, "1.0", (ml - 35, one_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    if len(bench_ret) == n:
        cv2.polylines(img, [_pts(bench_ret)],
                      isClosed=False, color=(140, 140, 140), thickness=1)
    cv2.polylines(img, [_pts(cum_ret)],
                  isClosed=False, color=(200, 60, 20), thickness=2)

    cv2.putText(img, "Backtest: Cumulative Returns",
                (ml, mt - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1)
    cv2.putText(img, "-- Strategy", (w - 240, mt + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 60, 20), 1)
    if len(bench_ret) == n:
        cv2.putText(img, "-- Benchmark", (w - 240, mt + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

    cv2.imwrite(str(save_path), img)
    log.info("回测曲线: %s", save_path)


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images) if not USE_AMP else autocast('cuda')(model)(images)
        all_logits.append(logits.cpu().numpy())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())
    return (np.array(all_preds), np.array(all_labels),
            np.concatenate(all_logits, axis=0))


def main():
    ckpt_path = CHECKPOINT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        log.error("未找到 %s", ckpt_path)
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    log.info("模型: %s (epoch=%d, val_f1=%.4f)",
             ckpt_path, ckpt.get("epoch", 0), ckpt.get("val_f1", 0))

    # ── 数据 ───────────────────────────────────────
    npz_files = sorted(SAMPLE_DIR.glob("*.npz"))
    if not npz_files:
        log.error("未找到 .npz 样本")
        return

    stock_list = sorted(set(f.stem for f in npz_files))
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.permutation(len(stock_list))
    n_train = int(len(stock_list) * 0.70)
    n_val   = int(len(stock_list) * 0.15)
    test_stocks = {stock_list[i] for i in idx[n_train + n_val:]}
    test_files = [f for f in npz_files if f.stem in test_stocks]

    test_ds = KLineDataset(test_files)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    log.info("测试集: %d 样本 (%d 只股票)", len(test_ds), len(test_stocks))

    # ── 推理 ───────────────────────────────────────
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    all_preds, all_labels, all_logits = predict_all(model, test_loader)
    all_probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()

    # ── 混淆矩阵 ───────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    log.info("\n══════════ 混淆矩阵 ══════════\n%s", cm)
    draw_confusion_matrix(cm, CLASS_NAMES, EVAL_DIR / "confusion_matrix.png")

    # ── 分类报告 ───────────────────────────────────
    report = classification_report(all_labels, all_preds,
                                   target_names=CLASS_NAMES, digits=4)
    log.info("\n══════════ 分类报告 ══════════\n%s", report)

    p, r, f1c, s = precision_recall_fscore_support(all_labels, all_preds,
                                                    labels=[0, 1])
    per_class_df = pd.DataFrame({
        "类别": CLASS_NAMES, "Precision": p, "Recall": r,
        "F1": f1c, "样本数": s,
    })
    log.info("\n%s", per_class_df.to_string(index=False))

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    log.info("\nAcc: %.4f  Macro-F1: %.4f  Weighted-F1: %.4f",
             acc, macro_f1, weighted_f1)

    # ── 真实回测 ───────────────────────────────────
    log.info("\n══════════ 真实回测 ══════════")
    price_map = load_price_data()
    log.info("加载 %d 只股票的价格数据", len(price_map))

    bt_results = run_real_backtest(
        all_preds, all_probs, test_ds.stocks, test_ds.dates,
        price_map, prob_threshold=PROB_THRESHOLD, holding_days=HOLDING_DAYS,
    )

    draw_backtest_curve(
        bt_results["cumulative_return"],
        bt_results["benchmark_return"],
        EVAL_DIR / "backtest_curve.png",
    )

    # ── 金融指标 ───────────────────────────────────
    m = bt_results["metrics"]
    log.info("\n══════════ 金融指标 ══════════")
    log.info("  交易次数:       %d", m.get("n_trades", 0))
    log.info("  总收益:         %+.2f%%", m.get("total_return", 0) * 100)
    log.info("  年化收益:       %+.2f%%", m.get("ann_return", 0) * 100)
    log.info("  年化波动:       %.2f%%", m.get("ann_volatility", 0) * 100)
    log.info("  Sharpe Ratio:   %.4f", m.get("sharpe_ratio", 0))
    log.info("  Sortino Ratio:  %.4f", m.get("sortino_ratio", 0))
    log.info("  Max Drawdown:   %.2f%%", m.get("max_drawdown", 0) * 100)
    log.info("  Calmar Ratio:   %.4f", m.get("calmar_ratio", 0))
    log.info("  Win Rate:       %.2f%%", m.get("win_rate", 0) * 100)
    log.info("  Profit Factor:  %.4f", m.get("profit_factor", 0))
    log.info("  Alpha:          %+.4f", m.get("alpha", 0))
    log.info("  Beta:           %.4f", m.get("beta", 0))

    # ── 按股票 ─────────────────────────────────────
    stock_perf = defaultdict(lambda: {"correct": 0, "total": 0})
    for stock, p, l in zip(test_ds.stocks, all_preds, all_labels):
        stock_perf[stock]["total"] += 1
        if p == l:
            stock_perf[stock]["correct"] += 1

    stock_acc = [{"stock": s, "accuracy": v["correct"] / v["total"],
                   "n": v["total"]}
                 for s, v in stock_perf.items()]
    stock_acc_df = pd.DataFrame(stock_acc).sort_values("accuracy", ascending=False)
    log.info("\n══════════ 按股票准确率 ══════════")
    log.info("Top 5:\n%s", stock_acc_df.head(5).to_string(index=False))
    log.info("Bottom 5:\n%s", stock_acc_df.tail(5).to_string(index=False))

    # ── 保存 ───────────────────────────────────────
    pred_df = pd.DataFrame({
        "stock": test_ds.stocks, "date": test_ds.dates,
        "true": all_labels, "pred": all_preds,
        "correct": (all_preds == all_labels).astype(int),
    })
    for i, name in enumerate(CLASS_NAMES):
        pred_df[f"prob_{name}"] = all_probs[:, i]
    pred_df.to_csv(EVAL_DIR / "predictions.csv", index=False)

    if len(bt_results["trade_log"]) > 0:
        bt_results["trade_log"].to_csv(EVAL_DIR / "trade_log.csv", index=False)

    with open(EVAL_DIR / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\nCV-K线趋势识别 评估报告\n" + "=" * 60 + "\n\n")
        line1 = "Accuracy:     {:.4f}\nMacro-F1:     {:.4f}\n".format(acc, macro_f1)
        line2 = "Weighted-F1:  {:.4f}\n\n".format(weighted_f1)
        f.write(line1 + line2)
        f.write("分类报告:\n" + report + "\n")
        f.write("各类别:\n" + per_class_df.to_string(index=False) + "\n\n")
        f.write("混淆矩阵:\n" + str(cm) + "\n\n")
        f.write("-- 回测指标 --\n")
        for k, v in m.items():
            if isinstance(v, float):
                f.write("  {}: {:.4f}\n".format(k, v))
            else:
                f.write("  {}: {}\n".format(k, v))
        f.write("\n按股票准确率:\n" + stock_acc_df.to_string(index=False) + "\n")

    log.info("\n完成。结果: %s", EVAL_DIR)


if __name__ == "__main__":
    main()
