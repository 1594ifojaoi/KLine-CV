"""
2_build_images.py
==================
从日线数据构建多通道 K 线图样本（OpenCV 绘制）。

六通道:
  Ch0 - K 线（蜡烛图，前复权）
  Ch1 - RSI (14)
  Ch2 - MACD (12/26/9)
  Ch3 - KDJ  (9,3,3)
  Ch4 - BOLL (20, 2σ)
  Ch5 - ATR  (14)

标签: 未来5日收益率 → {下跌, 上涨} (二分类, ±7%阈值, 丢弃中间震荡)

改进:
  - 修复 candle 实体 BUG（open/close 共用全局缩放）
  - 图像尺寸降为 128×128（60 根 K 线足够）
  - 使用前复权价格 (open_adj/high_adj/low_adj/close_adj)
"""

import os
import logging
import numpy as np
import pandas as pd
import cv2
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
SAMPLE_DIR  = BASE_DIR / "samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

# ── 图像参数 ──────────────────────────────────────────
IMG_SIZE   = 128          # 128×128（60 根 K 线足够，节省显存）
LOOKBACK   = 60           # 回看窗口（交易日）
FORECAST   = 5            # 预测未来 N 日
STRIDE     = 5            # 滑动步长
THRESHOLD  = 0.07         # 涨跌阈值（±7%，丢弃中间震荡样本）

# ── 指标参数 ──────────────────────────────────────────
RSI_PERIOD   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
KDJ_N        = 9
KDJ_M1       = 3
KDJ_M2       = 3
BOLL_PERIOD  = 20
BOLL_STD     = 2.0
ATR_PERIOD   = 14

LABEL_NAMES = {0: "下跌", 1: "上涨"}


# ═══════════════════════════════════════════════════════
#  指标计算（逐窗口计算，避免全序列 EMA 偏差）
# ═══════════════════════════════════════════════════════

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """指数移动平均（Wilder 式初始化: 首个值为 SMA）。"""
    result = np.full_like(series, np.nan, dtype=np.float64)
    if len(series) < span:
        return result
    result[span - 1] = series[:span].mean()
    alpha = 2.0 / (span + 1)
    for i in range(span, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
    return result


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI — Wilder 平滑。"""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = gain[1:period + 1].mean()
    avg_loss[period] = loss[1:period + 1].mean()

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_macd(close: np.ndarray,
                 fast: int = 12, slow: int = 26, signal: int = 9
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD — 返回 (DIF, DEA, MACD柱)。使用 EMA（非全序列）。"""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    dif = ema_fast - ema_slow
    dea = _ema(dif, signal) if len(dif) >= signal else np.full_like(dif, np.nan)
    macd_bar = 2.0 * (dif - dea)
    return dif, dea, macd_bar


def compute_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                n: int = 9, m1: int = 3, m2: int = 3
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """KDJ — K/D/J (EMA 平滑)。"""
    length = len(close)
    k = np.full(length, np.nan)
    d = np.full(length, np.nan)
    j = np.full(length, np.nan)
    if length < n:
        return k, d, j

    prev_k = 50.0
    prev_d = 50.0
    for i in range(n - 1, length):
        hh = high[i - n + 1:i + 1].max()
        ll = low[i - n + 1:i + 1].min()
        rsv = ((close[i] - ll) / (hh - ll)) * 100.0 if hh != ll else 50.0
        prev_k = (m1 - 1.0) / m1 * prev_k + 1.0 / m1 * rsv
        prev_d = (m2 - 1.0) / m2 * prev_d + 1.0 / m2 * prev_k
        k[i] = prev_k
        d[i] = prev_d
        j[i] = 3.0 * prev_k - 2.0 * prev_d
    return k, d, j


def compute_boll(close: np.ndarray,
                 period: int = 20, n_std: float = 2.0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BOLL — 上轨 / 中轨 / 下轨。"""
    mid = pd.Series(close).rolling(period, min_periods=period).mean().values
    std = pd.Series(close).rolling(period, min_periods=period).std(ddof=0).values
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """ATR — Wilder 平滑。"""
    n = len(close)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))

    atr[period] = tr[1:period + 1].mean()
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ═══════════════════════════════════════════════════════
#  OpenCV 绘图 — 修复版
# ═══════════════════════════════════════════════════════

def _make_scaler(values: np.ndarray, margin: float = 0.05):
    """
    返回一个闭包 scale_fn(v)，把数值映射到 [0, IMG_SIZE-1]。
    关键：用 values 全体确定 min/max，后续单点调用也共享这个范围。
    """
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return lambda x: np.full_like(np.asarray(x, dtype=float), IMG_SIZE // 2)

    v_min, v_max = v.min(), v.max()
    rng = v_max - v_min
    if rng == 0:
        return lambda x: np.full_like(np.asarray(x, dtype=float), IMG_SIZE // 2)

    lo = v_min - rng * margin
    hi = v_max + rng * margin

    def scale_fn(x):
        x = np.asarray(x, dtype=float)
        return (hi - x) / (hi - lo) * (IMG_SIZE - 1)
    return scale_fn


def _draw_line(img, x, y, color=255, thickness=1):
    """折线，跳过 NaN。"""
    valid = ~np.isnan(y)
    if valid.sum() < 2:
        return
    pts = np.column_stack([x[valid], y[valid]]).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=int(color), thickness=thickness)


def draw_channel_0(ohlc: np.ndarray) -> np.ndarray:
    """
    Ch0: K 线图。
    ★ 修复 BUG: 使用全局 scaler，所有 open/high/low/close 共享同一坐标范围。
    ohlc: shape (LOOKBACK, 4) → [open, high, low, close]  — 前复权
    """
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    n = len(ohlc)

    # ── 全局缩放（所有 price 共享） ──
    all_prices = ohlc.reshape(-1)
    scale = _make_scaler(all_prices)

    x_positions = np.linspace(2, IMG_SIZE - 3, n)

    for i in range(n):
        op, hi, lo, cl = ohlc[i]
        x = int(x_positions[i])

        y_hi = int(np.clip(scale(hi), 0, IMG_SIZE - 1))
        y_lo = int(np.clip(scale(lo), 0, IMG_SIZE - 1))
        y_op = int(np.clip(scale(op), 0, IMG_SIZE - 1))
        y_cl = int(np.clip(scale(cl), 0, IMG_SIZE - 1))

        # 影线
        cv2.line(img, (x, y_hi), (x, y_lo), 200, 1)

        # 实体
        top = min(y_op, y_cl)
        bot = max(y_op, y_cl)
        if bot - top < 1:
            bot = top + 1
        color = 255 if cl >= op else 80
        cv2.rectangle(img, (x - 1, top), (x + 1, bot), int(color), -1)

    return img


def draw_channel_1(values: np.ndarray) -> np.ndarray:
    """Ch1: RSI。"""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    n = len(values)
    x = np.linspace(2, IMG_SIZE - 3, n)

    # RSI 固定 0-100 范围
    scale = lambda v: (100.0 - np.asarray(v, float)) / 100.0 * (IMG_SIZE - 1)
    y = scale(values)

    # 参考线 30 / 50 / 70
    for level, c in [(30, 50), (50, 40), (70, 50)]:
        ly = int(np.clip(scale(level), 0, IMG_SIZE - 1))
        cv2.line(img, (2, ly), (IMG_SIZE - 3, ly), int(c), 1)

    _draw_line(img, x, y, color=255, thickness=1)
    return img


def draw_channel_2(dif, dea, macd_bar) -> np.ndarray:
    """Ch2: MACD。"""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    n = len(dif)
    x = np.linspace(2, IMG_SIZE - 3, n)

    combined = np.concatenate([dif, dea, macd_bar])
    scale = _make_scaler(combined, margin=0.05)
    baseline = scale(0.0)

    y_dif = scale(dif)
    y_dea = scale(dea)
    y_bar = scale(macd_bar)

    # 柱状图
    valid = ~np.isnan(y_bar)
    for i in np.where(valid)[0]:
        xi = int(x[i])
        yi = int(y_bar[i])
        bl = int(baseline)
        c = 200 if yi > bl else 100
        cv2.line(img, (xi, bl), (xi, yi), int(c), 1)

    _draw_line(img, x, y_dif, color=255, thickness=1)
    _draw_line(img, x, y_dea, color=150, thickness=1)

    zy = int(np.clip(baseline, 0, IMG_SIZE - 1))
    cv2.line(img, (2, zy), (IMG_SIZE - 3, zy), 50, 1)
    return img


def draw_channel_3(k, d, j) -> np.ndarray:
    """Ch3: KDJ — 固定 0-100 范围。"""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    n = len(k)
    x = np.linspace(2, IMG_SIZE - 3, n)

    scale = lambda v: (100.0 - np.asarray(v, float)) / 100.0 * (IMG_SIZE - 1)

    _draw_line(img, x, scale(k), color=255, thickness=1)
    _draw_line(img, x, scale(d), color=170, thickness=1)
    _draw_line(img, x, scale(j), color=85, thickness=1)

    for level, c in [(20, 50), (50, 40), (80, 50)]:
        ly = int(np.clip(scale(level), 0, IMG_SIZE - 1))
        cv2.line(img, (2, ly), (IMG_SIZE - 3, ly), int(c), 1)
    return img


def draw_channel_4(close, upper, mid, lower) -> np.ndarray:
    """Ch4: BOLL。"""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    n = len(close)
    x = np.linspace(2, IMG_SIZE - 3, n)

    # 用 BOLL 上下轨确定范围
    u_valid = upper[~np.isnan(upper)]
    l_valid = lower[~np.isnan(lower)]
    if len(u_valid) and len(l_valid):
        lo, hi = l_valid.min(), u_valid.max()
        rng = hi - lo or 1.0
        lo -= rng * 0.02
        hi += rng * 0.02
    else:
        c_valid = close[~np.isnan(close)]
        lo, hi = c_valid.min(), c_valid.max()
        rng = hi - lo or 1.0
        lo -= rng * 0.05
        hi += rng * 0.05

    def _s(v):
        return (hi - np.asarray(v, float)) / (hi - lo) * (IMG_SIZE - 1)

    _draw_line(img, x, _s(upper), color=100, thickness=1)
    _draw_line(img, x, _s(mid), color=170, thickness=1)
    _draw_line(img, x, _s(lower), color=100, thickness=1)
    _draw_line(img, x, _s(close), color=255, thickness=1)

    return img


def draw_channel_5(atr) -> np.ndarray:
    """Ch5: ATR。"""
    img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    n = len(atr)
    x = np.linspace(2, IMG_SIZE - 3, n)
    scale = _make_scaler(atr, margin=0.05)
    y = scale(atr)

    # 填充曲线下方
    valid = ~np.isnan(y)
    for i in np.where(valid)[0]:
        yi = int(np.clip(y[i], 0, IMG_SIZE - 1))
        cv2.line(img, (int(x[i]), yi), (int(x[i]), IMG_SIZE - 1), 60, 1)

    _draw_line(img, x, y, color=255, thickness=1)
    return img


# ═══════════════════════════════════════════════════════
#  样本构建
# ═══════════════════════════════════════════════════════

def build_samples_for_stock(csv_path: Path) -> dict | None:
    """为一只股票构建所有样本。使用前复权价格。"""
    ts_code = csv_path.stem.replace("_", ".")
    log.info("处理 %s ...", ts_code)

    try:
        df = pd.read_csv(csv_path, parse_dates=["trade_date"])
    except Exception as e:
        log.error("读取 %s 失败: %s", csv_path, e)
        return None

    df = df.sort_values("trade_date").reset_index(drop=True)

    # 优先使用前复权价格
    use_adj = all(c in df.columns for c in ["open_adj", "high_adj", "low_adj", "close_adj"])
    if use_adj:
        opn   = df["open_adj"].values.astype(np.float64)
        high  = df["high_adj"].values.astype(np.float64)
        low   = df["low_adj"].values.astype(np.float64)
        close = df["close_adj"].values.astype(np.float64)
        log.info("  -> 使用前复权价格")
    else:
        opn   = df["open"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)
        log.warning("  -> 无前复权列，使用原始价格")

    dates = df["trade_date"].dt.strftime("%Y-%m-%d").tolist()
    n_total = len(close)

    min_required = LOOKBACK + FORECAST
    if n_total < min_required:
        log.warning("%s 数据不足 (%d < %d)", ts_code, n_total, min_required)
        return None

    # ── 全序列计算指标 ──────────────────────────────
    rsi      = compute_rsi(close, RSI_PERIOD)
    dif, dea, macd_bar = compute_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    k, d, j  = compute_kdj(high, low, close, KDJ_N, KDJ_M1, KDJ_M2)
    upper, mid, lower = compute_boll(close, BOLL_PERIOD, BOLL_STD)
    atr      = compute_atr(high, low, close, ATR_PERIOD)

    # ── 滑动窗口 ────────────────────────────────────
    start_indices = list(range(0, n_total - min_required + 1, STRIDE))
    images, labels, window_dates = [], [], []

    for start in start_indices:
        end_img    = start + LOOKBACK
        end_future = end_img + FORECAST

        # 标签: 二分类，丢弃 ±7% 以内的震荡样本
        ret = (close[end_future - 1] - close[end_img - 1]) / close[end_img - 1]
        if ret > THRESHOLD:
            label = 1   # 上涨
        elif ret < -THRESHOLD:
            label = 0   # 下跌
        else:
            continue    # 震荡 → 丢弃

        # 切片
        sl = slice(start, end_img)
        ohlc = np.column_stack([opn[sl], high[sl], low[sl], close[sl]])

        ch0 = draw_channel_0(ohlc)
        ch1 = draw_channel_1(rsi[sl])
        ch2 = draw_channel_2(dif[sl], dea[sl], macd_bar[sl])
        ch3 = draw_channel_3(k[sl], d[sl], j[sl])
        ch4 = draw_channel_4(close[sl], upper[sl], mid[sl], lower[sl])
        ch5 = draw_channel_5(atr[sl])

        img_6ch = np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=-1).astype(np.uint8)
        images.append(img_6ch)
        labels.append(label)
        window_dates.append(dates[end_img - 1])

    if len(images) == 0:
        log.warning("%s 无有效窗口", ts_code)
        return None

    images_arr = np.stack(images, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)

    log.info("  -> %s: %d 样本, 跌=%d 涨=%d",
             ts_code, len(labels_arr),
             (labels_arr == 0).sum(),
             (labels_arr == 1).sum())

    return {
        "images": images_arr,
        "labels": labels_arr,
        "dates": window_dates,
        "ts_code": ts_code,
    }


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if f.stem != "manifest"]
    log.info("找到 %d 个数据文件", len(csv_files))

    all_images, all_labels, all_stocks, all_dates = [], [], [], []

    for csv_path in csv_files:
        result = build_samples_for_stock(csv_path)
        if result is None:
            continue

        safe_name = result["ts_code"].replace(".", "_")
        npz_path = SAMPLE_DIR / f"{safe_name}.npz"
        np.savez_compressed(
            npz_path,
            images=result["images"],
            labels=result["labels"],
            dates=np.array(result["dates"]),
            ts_code=result["ts_code"],
        )
        log.info("  保存 %s", npz_path)

        all_images.append(result["images"])
        all_labels.append(result["labels"])
        all_stocks.extend([result["ts_code"]] * len(result["labels"]))
        all_dates.extend(result["dates"])

    if not all_images:
        log.error("未生成任何样本！请先运行 1_download_data.py")
        return

    X = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)

    log.info("=" * 50)
    log.info("总计: %d 样本, 尺寸: %s, dtype: %s", len(y), X.shape, X.dtype)
    for lbl, name in LABEL_NAMES.items():
        log.info("  %s: %d (%.1f%%)", name, (y == lbl).sum(),
                 100 * (y == lbl).sum() / len(y))

    # 元数据
    meta = pd.DataFrame({
        "ts_code": all_stocks,
        "date": all_dates,
        "label": y,
    })
    meta.to_csv(SAMPLE_DIR / "metadata.csv", index=False)

    # 通道统计
    ch_means = X.mean(axis=(0, 1, 2))
    ch_stds  = X.std(axis=(0, 1, 2))
    stats = pd.DataFrame({"mean": ch_means, "std": ch_stds})
    stats.index.name = "channel"
    stats.to_csv(SAMPLE_DIR / "channel_stats.csv")
    log.info("通道统计:\n%s", stats)


if __name__ == "__main__":
    main()