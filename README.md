# KLine-CV

基于计算机视觉的 A 股 K 线趋势预测。将沪深 300 成分股的六项技术指标绘制为多通道图像，用轻量 CNN 识别未来 5 日涨跌方向。

## 原理

传统量化策略用数值特征（PE、换手率、均线值等）输入 MLP 或树模型。本项目换了一个思路：**把技术指标画成图**，让 CNN 像识别人脸一样识别 K 线形态中的涨跌信号。

| 通道 | 指标 | 参数 |
|------|------|------|
| Ch0 | K 线图（蜡烛图，前复权） | — |
| Ch1 | RSI | 14 周期 |
| Ch2 | MACD | 12/26/9 |
| Ch3 | KDJ | 9/3/3 |
| Ch4 | BOLL | 20 周期, 2σ |
| Ch5 | ATR | 14 周期 |

每张图覆盖 60 个交易日（128×128 像素），标签为未来 5 日是否涨/跌超过 ±7%。

## 模型

- **架构**：4 层轻量 CNN，约 61K 参数
- **优化器**：SAM (Sharpness-Aware Minimization, ρ=0.05) + AdamW
- **正则化**：Dropout(0.5) + weight_decay(1e-3) + Label Smoothing(0.02)
- **数据划分**：股票级别 70/15/15 分割，杜绝同股跨集合泄露

## 结果

| 指标 | 数值 |
|------|------|
| Test F1 | 0.563 |
| Win Rate | 59% |
| Profit Factor | 1.75 |
| Train-Val F1 差距 | ~0.03（零过拟合） |

## 快速开始

### 环境

- Python 3.10+
- PyTorch 2.x (CUDA 可选)
- Tushare Pro Token（[注册获取](https://tushare.pro)）

```bash
pip install torch torchvision pandas numpy opencv-python tushare scikit-learn -U
```

### 运行

```bash
# 0. 设置 Tushare Token（仅需一次）
# Windows PowerShell:
$env:TUSHARE_TOKEN = "你的token"
# Linux/Mac:
export TUSHARE_TOKEN="你的token"

# 1. 下载沪深300成分股日线数据（2015-2025）
python 1_download_data.py

# 2. 生成六通道K线图样本
python 2_build_images.py

# 3. 训练
python 3_train_resnet.py

# 4. 评估 + 回测
python 4_evaluate.py
```

## 目录结构

```
cv_K/
├── 1_download_data.py    # 数据下载
├── 2_build_images.py     # 图像样本生成
├── 3_train_resnet.py     # 模型训练
├── 4_evaluate.py         # 评估与回测
├── data/                 # 原始CSV（gitignore）
├── samples/              # 图像样本（gitignore）
├── checkpoints/          # 模型权重（gitignore）
├── logs/                 # 训练日志
├── eval_results/         # 评估报告与回测曲线
└── requirements.txt
```

## 关键设计决策

- **二分类优于三分类**：丢弃 ±7% 以内的震荡样本，模型不再被大量无区分度的中间态干扰
- **小模型优于大模型**：61K 参数 vs ResNet18 的 1100 万，防止在高度重叠的时序样本上过拟合
- **股票级别划分**：同一股票的所有时间窗口分配至同一集合，杜绝数据泄露
- **SAM 优化**：引导权重收敛至 loss 曲面的平坦区域，泛化显著优于普通 SGD/AdamW

## License

MIT
