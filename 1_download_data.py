"""
1_download_data.py
==================
下载沪深300全部成分股 2015-2025 年日线数据（含复权因子）。

改进:
  - 新增 adj_factor → 前复权 OHLCV
  - 下载全部 HS300（不再随机抽样，保证可复现）
  - 逐只保存为 CSV
"""

import os
import time
import logging
import pandas as pd
import numpy as np
import tushare as ts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── 配置 ──────────────────────────────────────────────
TOKEN = os.environ.get("TUSHARE_TOKEN", "")
if not TOKEN:
    raise RuntimeError("请设置环境变量 TUSHARE_TOKEN，例如: set TUSHARE_TOKEN=你的token")
START_DATE = "20150101"
END_DATE   = "20251231"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Tushare 初始化 ────────────────────────────────────
pro = ts.pro_api(TOKEN)
log.info("Tushare API configured (official endpoint)")


def get_hs300_stocks() -> list[str]:
    """获取沪深300成分股 ts_code 列表。"""
    methods = []
    codes = None

    # 方法1: hs300 接口
    try:
        df = pro.hs300()
        if df is not None and len(df) > 0:
            codes = df["ts_code"].tolist()
            methods.append("hs300")
    except Exception as e:
        log.warning("hs300 接口失败: %s", e)

    # 方法2: index_weight 接口
    if codes is None:
        try:
            df = pro.index_weight(
                index_code="000300.SH",
                start_date=START_DATE,
                end_date=END_DATE,
            )
            if df is not None and len(df) > 0:
                codes = df["con_code"].drop_duplicates().tolist()
                methods.append("index_weight")
        except Exception as e:
            log.warning("index_weight 接口失败: %s", e)

    if codes is None or len(codes) == 0:
        raise RuntimeError("无法获取沪深300成分股，请检查网络或Token。")

    log.info("获取沪深300成分股 %d 只 (via %s)", len(codes), ", ".join(methods))
    return codes


def download_with_retry(fn, desc: str, max_retries: int = 3):
    """带重试的下载包装器。"""
    for attempt in range(max_retries):
        try:
            result = fn()
            if result is not None and len(result) > 0:
                return result
        except Exception as e:
            log.warning("%s 失败 (attempt %d/%d): %s", desc, attempt + 1, max_retries, e)
            time.sleep(1.5)
    return None


def download_one_stock(ts_code: str) -> pd.DataFrame | None:
    """
    下载单只股票的日线 + 复权因子，合成前复权 OHLCV。

    返回列:
      trade_date, open, high, low, close, vol, amount,
      open_adj, high_adj, low_adj, close_adj
    """

    # ── daily ──────────────────────────────────────
    df_daily = download_with_retry(
        lambda: pro.daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE),
        f"{ts_code} daily",
    )
    if df_daily is None:
        return None

    df_daily["trade_date"] = pd.to_datetime(df_daily["trade_date"], format="%Y%m%d")
    df_daily = df_daily.sort_values("trade_date").reset_index(drop=True)

    # ── adj_factor ─────────────────────────────────
    df_adj = download_with_retry(
        lambda: pro.adj_factor(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE),
        f"{ts_code} adj_factor",
    )

    if df_adj is not None:
        df_adj["trade_date"] = pd.to_datetime(df_adj["trade_date"], format="%Y%m%d")
        df = df_daily.merge(df_adj, on=["ts_code", "trade_date"], how="left")

        # 前复权因子: adj_factor 已经是前复权因子
        # 前复权价格 = 原始价格 × adj_factor
        if "adj_factor" in df.columns:
            factor = df["adj_factor"].fillna(1.0).values
            df["open_adj"]  = df["open"].values  * factor
            df["high_adj"]  = df["high"].values  * factor
            df["low_adj"]   = df["low"].values   * factor
            df["close_adj"] = df["close"].values * factor
        else:
            log.warning("%s adj_factor 列缺失，使用原始价格", ts_code)
            for col in ["open", "high", "low", "close"]:
                df[f"{col}_adj"] = df[col]
    else:
        log.warning("%s 无复权因子数据，使用原始价格", ts_code)
        for col in ["open", "high", "low", "close"]:
            df[f"{col}_adj"] = df[col]

    # ── daily_basic (换手率等辅助指标) ─────────────
    df_basic = download_with_retry(
        lambda: pro.daily_basic(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE),
        f"{ts_code} daily_basic",
    )
    if df_basic is not None:
        df_basic["trade_date"] = pd.to_datetime(df_basic["trade_date"], format="%Y%m%d")
        # 只合并换手率等核心字段，避免列冲突
        keep_cols = ["trade_date"]
        for c in ["turnover_rate", "turnover_rate_f", "volume_ratio", "pe", "pe_ttm", "pb"]:
            if c in df_basic.columns:
                keep_cols.append(c)
        df = df.merge(df_basic[keep_cols], on="trade_date", how="left")

    return df


def main():
    # 1. 获取全部 HS300 成分股
    all_codes = get_hs300_stocks()
    log.info("共 %d 只股票待下载", len(all_codes))

    success = 0
    fail_list = []

    for i, ts_code in enumerate(all_codes, 1):
        log.info("[%03d/%03d] %s", i, len(all_codes), ts_code)

        df = download_one_stock(ts_code)
        if df is None or len(df) == 0:
            log.warning("  -> %s 下载失败，跳过", ts_code)
            fail_list.append(ts_code)
            continue

        # 保存
        safe_name = ts_code.replace(".", "_")
        csv_path = os.path.join(DATA_DIR, f"{safe_name}.csv")
        df.to_csv(csv_path, index=False)
        success += 1
        log.info("  -> %d 条记录 → %s", len(df), csv_path)

        time.sleep(0.5)  # 频率控制

    # 2. 汇总
    log.info("=" * 50)
    log.info("完成! 成功 %d / %d 只", success, len(all_codes))
    if fail_list:
        log.warning("失败 %d 只: %s", len(fail_list), fail_list)

    # 写入选股清单
    manifest_path = os.path.join(DATA_DIR, "manifest.txt")
    downloaded = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    with open(manifest_path, "w") as f:
        f.write("\n".join(sorted(downloaded)))
    log.info("清单写入 %s (%d 个文件)", manifest_path, len(downloaded))


if __name__ == "__main__":
    main()