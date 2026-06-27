"""
=============================================================================
  A股+美股 多因子量化分析系统  —  FastAPI + AkShare
  单票分析 + 多票对比 + 动态权重 + 概率评估
=============================================================================
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="A股+美股多因子量化分析", version="2.0.0")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ── 缓存 ───────────────────────────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}
CACHE_TTL = 600  # 10分钟（日线数据交易日不变）


def _cache_key(prefix: str, code: str) -> str:
    today = datetime.now().strftime("%Y%m%d")
    return f"{prefix}:{code}:{today}"


def _cache_get(key: str):
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return val
        del _cache[key]
    return None


def _cache_set(key: str, val):
    _cache[key] = (time.time(), val)
    now = time.time()
    for k in list(_cache.keys()):
        if now - _cache[k][0] > CACHE_TTL * 3:
            del _cache[k]


# ── 市场检测 ────────────────────────────────────────────────

def detect_market(code: str) -> str:
    """检测市场: 'cn' = A股, 'us' = 美股"""
    code = code.strip().upper()
    if re.match(r'^\d{6}$', code):
        return 'cn'
    if re.match(r'^[A-Z]{1,5}$', code):
        return 'us'
    raise ValueError(f"无法识别市场: {code}，请输入6位A股代码或美股ticker（如 AAPL）")


# ── 数据获取（A股）──────────────────────────────────────────

def _fetch_daily(code: str, days: int = 250) -> pd.DataFrame:
    """获取日K线（前复权）—— 单票接口，2-5秒"""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start,
                            end_date=end, adjust="qfq")
    if df is None or df.empty:
        raise ValueError(f"无法获取 {code} 的日K线数据")
    df = df.sort_values("日期").tail(days).reset_index(drop=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def _get_daily(code: str, days: int = 250) -> pd.DataFrame:
    key = _cache_key("daily", code)
    cached = _cache_get(key)
    if cached is not None:
        if len(cached) >= days:
            return cached.tail(days)
    df = _fetch_daily(code, days)
    _cache_set(key, df)
    return df.tail(days)


def _get_realtime(code: str, daily_df: pd.DataFrame) -> dict:
    """从日K线提取实时行情（无需全市场接口，秒出）"""
    row = daily_df.iloc[-1]
    return {
        "代码": code,
        "名称": code,
        "最新价": float(row["收盘"]),
        "涨跌幅": float(row.get("涨跌幅", 0)),
        "换手率": float(row.get("换手率", 0)),
        "总市值": 0,  # 由财务数据补充
        "市盈率-动态": 0,
        "成交量": float(row.get("成交量", 0)),
        "成交额": float(row.get("成交额", 0)),
    }


def _get_financials(code: str) -> dict:
    """获取基本面数据（使用 stock_financial_abstract）"""
    key = _cache_key("fin", code)
    cached = _cache_get(key)
    if cached is not None:
        records = cached.to_dict("records")
        if records:
            return records[0]
        return {}
    try:
        fin = ak.stock_financial_abstract(symbol=code)
        if fin is None or fin.empty:
            raise ValueError("empty financial data")
        # 提取关键指标：指标名 → 最新季度值
        result = {}
        for i in range(len(fin)):
            metric = str(fin.iloc[i, 1]).strip()
            val = fin.iloc[i, 2]  # 最新季度（第3列是最近期）
            if pd.notna(val):
                try:
                    result[metric] = float(val)
                except (ValueError, TypeError):
                    result[metric] = val
        _cache_set(key, pd.DataFrame([result]))
        return result
    except Exception:
        pass
    empty = {}
    _cache_set(key, pd.DataFrame([empty]))
    return empty


# ── 数据获取（美股）──────────────────────────────────────────

_US_COLUMN_MAP = {
    "date": "日期", "open": "开盘", "high": "最高", "low": "最低",
    "close": "收盘", "volume": "成交量",
}


def _normalize_us_df(raw: pd.DataFrame) -> pd.DataFrame:
    """美股 DataFrame → 统一列名（与A股因子兼容）"""
    df = raw.copy()
    df = df.rename(columns={k: v for k, v in _US_COLUMN_MAP.items() if k in df.columns})
    # 补充A股因子需要的列
    if "成交额" not in df.columns:
        df["成交额"] = 0
    if "换手率" not in df.columns:
        df["换手率"] = 0
    if "涨跌幅" not in df.columns and "收盘" in df.columns:
        df["涨跌幅"] = df["收盘"].pct_change() * 100
    if "涨跌额" not in df.columns:
        df["涨跌额"] = 0
    if "振幅" not in df.columns and "最高" in df.columns and "最低" in df.columns:
        df["振幅"] = (df["最高"] - df["最低"]) / df["收盘"].shift(1) * 100
    return df


def _fetch_daily_us(ticker: str, days: int = 250) -> pd.DataFrame:
    """获取美股日K线（前复权）"""
    df = ak.stock_us_daily(symbol=ticker, adjust="qfq")
    if df is None or df.empty:
        raise ValueError(f"无法获取美股 {ticker} 的日K线数据")
    df = df.sort_values("date").tail(days).reset_index(drop=True)
    return _normalize_us_df(df)


def _get_daily_us(ticker: str, days: int = 250) -> pd.DataFrame:
    key = _cache_key("daily_us", ticker)
    cached = _cache_get(key)
    if cached is not None:
        if len(cached) >= days:
            return cached.tail(days)
    df = _fetch_daily_us(ticker, days)
    _cache_set(key, df)
    return df.tail(days)


def _get_realtime_us(ticker: str, daily_df: pd.DataFrame) -> dict:
    """从美股日K线提取实时行情"""
    row = daily_df.iloc[-1]
    price = float(row["收盘"])
    prev = float(daily_df["收盘"].iloc[-2]) if len(daily_df) >= 2 else price
    return {
        "代码": ticker,
        "名称": ticker,  # 后面由财务数据补充
        "最新价": price,
        "涨跌幅": round((price / prev - 1) * 100, 2),
        "换手率": 0,  # 美股日K线无换手率
        "总市值": 0,
        "市盈率-动态": 0,
        "成交量": float(row.get("成交量", 0)),
        "成交额": float(row.get("成交额", 0)),
        "内盘": 0,
        "外盘": 0,
    }


def _get_financials_us(ticker: str) -> dict:
    """获取美股基本面数据"""
    key = _cache_key("fin_us", ticker)
    cached = _cache_get(key)
    if cached is not None:
        records = cached.to_dict("records")
        if records:
            return records[0]
        return {}
    try:
        fin = ak.stock_financial_us_analysis_indicator_em(symbol=ticker)
        if fin is None or fin.empty:
            raise ValueError("empty")
        latest = fin.iloc[-1].to_dict()
        result = {}
        # 映射到中文名（与A股因子兼容）
        field_map = {
            "SECURITY_NAME_ABBR": "名称",
            "BASIC_EPS": "基本每股收益",
            "DILUTED_EPS": "摊薄每股收益_最新股本",
            "PARENT_HOLDER_NETPROFIT": "归母净利润",
            "OPERATE_INCOME": "营业总收入",
            "ROE_AVG": "净资产收益率",
            "GROSS_PROFIT_RATIO": "毛利率",
            "NET_PROFIT_RATIO": "净利润率",
        }
        for en, cn in field_map.items():
            val = latest.get(en)
            if pd.notna(val):
                try:
                    result[cn] = float(val)
                except (ValueError, TypeError):
                    result[cn] = val
        # 保留原名
        name = latest.get("SECURITY_NAME_ABBR")
        if name and pd.notna(name):
            result["_display_name"] = str(name)
        _cache_set(key, pd.DataFrame([result]))
        return result
    except Exception:
        pass
    empty = {}
    _cache_set(key, pd.DataFrame([empty]))
    return empty


# ── 统一数据获取入口 ─────────────────────────────────────────

def _get_daily_unified(code: str, market: str, days: int = 250) -> pd.DataFrame:
    if market == 'us':
        return _get_daily_us(code, days)
    return _get_daily(code, days)


def _get_realtime_unified(code: str, market: str, df: pd.DataFrame) -> dict:
    if market == 'us':
        return _get_realtime_us(code, df)
    return _get_realtime(code, df)


def _get_financials_unified(code: str, market: str) -> dict:
    if market == 'us':
        return _get_financials_us(code)
    return _get_financials(code)


# ── 指数数据 ───────────────────────────────────────────────

def _get_index_daily() -> pd.DataFrame:
    """获取上证指数日K线"""
    key = _cache_key("idx", "sh000001")
    cached = _cache_get(key)
    if cached is not None:
        return cached
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=400)).strftime("%Y%m%d")
    df = ak.stock_zh_index_daily_em(symbol="sh000001", start_date=start, end_date=end)
    df = df.sort_values("date").tail(120).reset_index(drop=True)
    _cache_set(key, df)
    return df


def _get_sp500_index() -> pd.DataFrame:
    """获取标普500日K线（SPY作为代理）"""
    key = _cache_key("idx", "sp500")
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        raw = ak.stock_us_daily(symbol="SPY", adjust="qfq")
        df = raw.sort_values("date").tail(120).reset_index(drop=True)
        _cache_set(key, df)
        return df
    except Exception:
        return pd.DataFrame()


@lru_cache(maxsize=1)
def _get_stock_list_df() -> pd.DataFrame:
    try:
        return ak.stock_info_a_code_name()
    except Exception:
        return pd.DataFrame()


# ── 搜索 ────────────────────────────────────────────────────

# 常用美股映射
_US_POPULAR: dict[str, str] = {
    "AAPL": "苹果", "MSFT": "微软", "GOOGL": "谷歌", "AMZN": "亚马逊",
    "NVDA": "英伟达", "META": "Meta", "TSLA": "特斯拉", "BRK.B": "伯克希尔",
    "JPM": "摩根大通", "V": "Visa", "JNJ": "强生", "WMT": "沃尔玛",
    "MA": "万事达", "PG": "宝洁", "UNH": "联合健康", "HD": "家得宝",
    "DIS": "迪士尼", "NFLX": "奈飞", "ADBE": "Adobe", "CRM": "Salesforce",
    "AMD": "AMD", "INTC": "英特尔", "QCOM": "高通", "TXN": "德州仪器",
    "AVGO": "博通", "ORCL": "甲骨文", "IBM": "IBM", "CSCO": "思科",
    "PYPL": "PayPal", "UBER": "优步", "ABNB": "爱彼迎", "SNAP": "Snap",
    "COIN": "Coinbase", "BA": "波音", "CAT": "卡特彼勒", "GE": "通用电气",
    "XOM": "埃克森美孚", "CVX": "雪佛龙", "PFE": "辉瑞", "MRK": "默沙东",
    "COST": "好市多", "NKE": "耐克", "SBUX": "星巴克", "MCD": "麦当劳",
    "BABA": "阿里巴巴", "JD": "京东", "PDD": "拼多多", "NIO": "蔚来",
    "BIDU": "百度", "TCOM": "携程", "BILI": "哔哩哔哩", "XPEV": "小鹏",
    "LI": "理想汽车", "FUTU": "富途", "BEKE": "贝壳",
    "PLTR": "Palantir", "SNOW": "Snowflake", "SHOP": "Shopify",
    "SPY": "标普500ETF", "QQQ": "纳斯达克100ETF",
}

# 英文别名 → ticker（用于搜索）
_US_ALIAS: dict[str, str] = {
    "APPLE": "AAPL", "MICROSOFT": "MSFT", "GOOGLE": "GOOGL", "AMAZON": "AMZN",
    "NVIDIA": "NVDA", "TESLA": "TSLA", "META": "META", "FACEBOOK": "META",
    "BERKSHIRE": "BRK.B", "JPMORGAN": "JPM", "VISA": "V",
    "JOHNSON": "JNJ", "WALMART": "WMT", "MASTERCARD": "MA",
    "PROCTER": "PG", "DISNEY": "DIS", "NETFLIX": "NFLX",
    "ADOBE": "ADBE", "SALESFORCE": "CRM", "INTEL": "INTC",
    "QUALCOMM": "QCOM", "BROADCOM": "AVGO", "ORACLE": "ORCL",
    "CISCO": "CSCO", "PAYPAL": "PYPL", "UBER": "UBER",
    "AIRBNB": "ABNB", "COINBASE": "COIN", "BOEING": "BA",
    "CATERPILLAR": "CAT", "EXXON": "XOM", "CHEVRON": "CVX",
    "PFIZER": "PFE", "MERCK": "MRK", "COSTCO": "COST",
    "NIKE": "NKE", "STARBUCKS": "SBUX", "MCDONALD": "MCD",
    "ALIBABA": "BABA", "PINDUODUO": "PDD", "NIO": "NIO",
    "BAIDU": "BIDU", "BILIBILI": "BILI", "XPENG": "XPEV",
    "LI AUTO": "LI", "LIAUTO": "LI", "FUTU": "FUTU",
    "BEIKE": "BEKE", "KE": "BEKE", "PALANTIR": "PLTR",
    "SNOWFLAKE": "SNOW", "SHOPIFY": "SHOP",
}


@lru_cache(maxsize=1)
def _get_stock_list_df() -> pd.DataFrame:
    try:
        return ak.stock_info_a_code_name()
    except Exception:
        return pd.DataFrame()


def search_stock(name: str, limit: int = 10) -> list[dict]:
    """搜索股票 — 支持A股名称和美股ticker"""
    name_upper = name.strip().upper()
    results = []

    # 美股 ticker 精确/前缀匹配
    if re.match(r'^[A-Z]{1,5}$', name_upper):
        if name_upper in _US_POPULAR:
            results.append({"代码": name_upper, "名称": _US_POPULAR[name_upper], "市场": "美股"})
        for ticker, cname in _US_POPULAR.items():
            if ticker.startswith(name_upper) and ticker != name_upper:
                results.append({"代码": ticker, "名称": cname, "市场": "美股"})
            if len(results) >= limit:
                return results[:limit]

        # 英文别名匹配
        ticker_from_alias = _US_ALIAS.get(name_upper)
        if ticker_from_alias and ticker_from_alias in _US_POPULAR:
            entry = {"代码": ticker_from_alias, "名称": _US_POPULAR[ticker_from_alias], "市场": "美股"}
            if entry not in results:
                results.append(entry)

        return results[:limit]

    # 非纯代码 → 模糊搜索（美股+ A股）
    # 美股中文名/英文名匹配
    for ticker, cname in _US_POPULAR.items():
        if name.upper() in cname.upper() or name.upper() in ticker:
            entry = {"代码": ticker, "名称": cname, "市场": "美股"}
            if entry not in results:
                results.append(entry)
        if len(results) >= limit:
            return results[:limit]

    # 英文别名匹配
    for alias, ticker in _US_ALIAS.items():
        if name_upper in alias or alias in name_upper:
            ticker_cn = _US_POPULAR.get(ticker, ticker)
            entry = {"代码": ticker, "名称": ticker_cn, "市场": "美股"}
            if entry not in results:
                results.append(entry)
        if len(results) >= limit:
            return results[:limit]

    # A股搜索（仅中文查询时启用）
    looks_chinese = bool(re.search(r'[\u4e00-\u9fff]', name))
    if len(results) < limit and looks_chinese:
        try:
            df = _get_stock_list_df()
            if not df.empty:
                mask = df["名称"].str.contains(name, na=False)
                for _, row in df[mask].head(limit - len(results)).iterrows():
                    results.append({"代码": row["代码"], "名称": row["名称"], "市场": "A股"})
        except Exception:
            pass

    return results[:limit]


# ── 技术指标计算 ────────────────────────────────────────────

def calc_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def calc_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def calc_macd(close: pd.Series):
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    dif = ema12 - ema26
    dea = calc_ema(dif, 9)
    macd_bar = 2 * (dif - dea)
    return dif, dea, macd_bar


def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_bollinger(close: pd.Series, window: int = 20):
    mid = calc_ma(close, window)
    std = close.rolling(window=window).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    return upper, mid, lower


def calc_kdj(high, low, close, n=9):
    low_n = low.rolling(window=n).min()
    high_n = high.rolling(window=n).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


# ── 数据框辅助 ──────────────────────────────────────────────

def _safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    for c in df.columns:
        if c.strip() == col.strip():
            return df[c]
    raise KeyError(f"列 '{col}' 不存在。可用列: {list(df.columns)}")


def _col(df: pd.DataFrame, col: str) -> pd.Series:
    return _safe_col(df, col)


# ── 因子评分 ────────────────────────────────────────────────

def _percentile_score(value: float, series: pd.Series, reverse: bool = False) -> float:
    """将值在历史序列中的分位数映射到 1-5 分"""
    if len(series) < 10:
        return 3.0
    pct = (series < value).sum() / len(series)
    if reverse:
        pct = 1 - pct
    return round(1 + pct * 4, 2)


def factor_momentum(df: pd.DataFrame, realtime: dict) -> dict:
    """动量因子"""
    close = _col(df, "收盘")
    if len(close) < 10:
        return {"score": 3.0, "detail": {}}

    def _ret(days):
        if len(close) <= days:
            return None
        return round((close.iloc[-1] / close.iloc[-(days+1)] - 1) * 100, 2)

    returns = {"5日": _ret(5), "10日": _ret(10), "20日": _ret(20), "60日": _ret(60)}
    if len(close) >= 250:
        returns["YTD"] = round((close.iloc[-1] / close.iloc[-250] - 1) * 100, 2) if len(close) >= 250 else None

    # 用5日和20日动量算分
    valid = {k: v for k, v in returns.items() if v is not None}
    if not valid:
        return {"score": 3.0, "detail": returns}

    # 综合：短期动量 + 中期动量
    short = returns.get("5日", 0) or 0
    mid = returns.get("20日", 0) or 0
    long = returns.get("60日", 0) or 0

    # 评分逻辑
    raw = 0
    if short > 3: raw += 1.5
    elif short > 0: raw += 1.0
    elif short > -3: raw += 0.5

    if mid > 5: raw += 1.5
    elif mid > 0: raw += 1.0
    elif mid > -5: raw += 0.5

    if long > 10: raw += 1.5
    elif long > 0: raw += 1.0
    elif long > -10: raw += 0.5

    score = min(5.0, max(1.0, round(raw + 0.5, 2)))

    # 均线排列加分
    above_count = 0
    try:
        ma_list = [5, 10, 20, 30, 60]
        mas = {m: calc_ma(close, m).iloc[-1] for m in ma_list}
        price = float(close.iloc[-1])
        above_count = sum(1 for m in ma_list if price > mas[m])
        if above_count == 5:
            score = min(5.0, score + 0.5)
        elif above_count >= 3:
            score = min(5.0, score + 0.2)
    except Exception:
        pass

    score = round(score, 2)
    return {"score": score, "detail": returns, "above_ma": above_count if "above_count" in dir() else None}


def factor_reversion(df: pd.DataFrame) -> dict:
    """均值回归因子"""
    close = _col(df, "收盘")
    if len(close) < 60:
        return {"score": 3.0, "detail": {}}

    price = float(close.iloc[-1])
    high_52w = float(close.iloc[-60:].max())
    low_52w = float(close.iloc[-60:].min())

    drop_from_high = (1 - price / high_52w) * 100 if high_52w > 0 else 0

    # 布林带
    upper, mid_bb, lower = calc_bollinger(close)
    bb_pos = (price - float(lower.iloc[-1])) / (float(upper.iloc[-1]) - float(lower.iloc[-1])) if float(upper.iloc[-1]) != float(lower.iloc[-1]) else 0.5

    # 评分：超卖（跌幅大）→ 高分（回归概率高）
    if drop_from_high > 30:
        score = 4.0
    elif drop_from_high > 20:
        score = 3.5
    elif drop_from_high > 10:
        score = 3.0
    elif drop_from_high > 5:
        score = 2.5
    elif drop_from_high > 0:
        score = 2.0
    else:
        score = 1.5  # 历史新高附近，回归向下

    return {
        "score": round(min(5.0, max(1.0, score)), 2),
        "detail": {
            "52周跌幅": round(drop_from_high, 1),
            "52周最高": round(high_52w, 2),
            "52周最低": round(low_52w, 2),
            "布林带位置": round(bb_pos * 100, 1),
        }
    }


def factor_technical(df: pd.DataFrame) -> dict:
    """技术指标因子"""
    close = _col(df, "收盘")
    high = _col(df, "最高")
    low = _col(df, "最低")

    if len(close) < 30:
        return {"score": 3.0, "detail": {}}

    # MACD
    dif, dea, macd_bar = calc_macd(close)
    dif_v, dea_v = float(dif.iloc[-1]), float(dea.iloc[-1])
    macd_signal = "金叉" if dif_v > dea_v else "死叉"
    macd_diff = round(dif_v - dea_v, 4)

    # RSI
    rsi_6 = calc_rsi(close, 6)
    rsi_12 = calc_rsi(close, 12)
    rsi6_v = float(rsi_6.iloc[-1])

    # KDJ
    k, d, j = calc_kdj(high, low, close)
    k_v, d_v, j_v = float(k.iloc[-1]), float(d.iloc[-1]), float(j.iloc[-1])
    kdj_signal = "金叉" if k_v > d_v else "死叉"

    # 布林带
    upper, mid_bb, lower = calc_bollinger(close)
    price = float(close.iloc[-1])
    bb_pos = (price - float(lower.iloc[-1])) / (float(upper.iloc[-1]) - float(lower.iloc[-1]))

    # 综合评分
    score = 3.0

    # MACD
    if macd_diff > 0.1: score += 0.5
    elif macd_diff > 0: score += 0.2
    elif macd_diff > -0.1: score -= 0.2
    else: score -= 0.5

    # RSI
    if 40 <= rsi6_v <= 60: score += 0.3
    elif 30 <= rsi6_v < 40: score += 0.5  # 超卖区
    elif rsi6_v > 70: score -= 0.3

    # KDJ
    if k_v > d_v: score += 0.2
    if j_v > 100: score -= 0.3

    # 布林带位置
    if 0.3 < bb_pos < 0.7: score += 0.2

    score = round(min(5.0, max(1.0, score)), 2)

    return {
        "score": score,
        "detail": {
            "MACD": {"DIF": round(dif_v, 3), "DEA": round(dea_v, 3), "信号": macd_signal},
            "RSI": {"RSI6": round(rsi6_v, 1), "RSI12": round(float(rsi_12.iloc[-1]), 1)},
            "KDJ": {"K": round(k_v, 1), "D": round(d_v, 1), "J": round(j_v, 1), "信号": kdj_signal},
            "布林带": {"上轨": round(float(upper.iloc[-1]), 2), "中轨": round(float(mid_bb.iloc[-1]), 2), "下轨": round(float(lower.iloc[-1]), 2)},
        }
    }


def factor_volume(df: pd.DataFrame, realtime: dict) -> dict:
    """量价因子"""
    close = _col(df, "收盘")
    try:
        volume = _col(df, "成交量")
    except KeyError:
        volume = None
    try:
        turnover = _col(df, "换手率")
    except KeyError:
        turnover = None

    if len(close) < 20:
        return {"score": 3.0, "detail": {}}

    # 换手率
    try:
        t_rate = float(realtime.get("换手率", 0))
    except (ValueError, TypeError):
        t_rate = float(turnover.iloc[-1]) if turnover is not None and len(turnover) > 0 else 0

    # 量比
    try:
        vol_ratio = float(realtime.get("量比", 1))
    except (ValueError, TypeError):
        vol_ratio = 1.0

    # 价格涨跌
    try:
        change_pct = float(realtime.get("涨跌幅", 0))
    except (ValueError, TypeError):
        if len(close) >= 2:
            change_pct = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100
        else:
            change_pct = 0

    # 内盘/外盘
    try:
        inner = float(realtime.get("内盘", 0))
        outer = float(realtime.get("外盘", 0))
    except (ValueError, TypeError):
        inner, outer = 0, 0

    score = 3.0

    # 量价配合
    if change_pct > 3 and t_rate > 5 and vol_ratio > 1.5:
        score += 1.0  # 放量上涨
    elif change_pct > 1 and t_rate > 3 and vol_ratio > 1.2:
        score += 0.5
    elif change_pct < -3 and t_rate > 5:
        score -= 1.0  # 放量下跌
    elif change_pct < -1 and t_rate > 3:
        score -= 0.5

    # 换手率异常
    if t_rate > 20:
        score -= 0.5  # 天量换手，警惕
    elif t_rate > 10:
        score -= 0.2

    # 内外盘
    if outer > inner and inner > 0:
        score += 0.2
    elif inner > outer and outer > 0:
        score -= 0.2

    score = round(min(5.0, max(1.0, score)), 2)

    return {
        "score": score,
        "detail": {
            "换手率": round(t_rate, 2),
            "量比": round(vol_ratio, 2),
            "涨跌幅": round(change_pct, 2),
            "内盘占比": round(inner / (inner + outer) * 100, 1) if (inner + outer) > 0 else 50,
        }
    }


def factor_fundamental(fin: dict, realtime: dict, df: pd.DataFrame) -> dict:
    """基本面因子"""
    close = _col(df, "收盘")
    price = float(close.iloc[-1])

    # PE
    pe = None
    try:
        pe = float(realtime.get("市盈率-动态", 0) or realtime.get("市盈率（动态）", 0))
        if pe <= 0: pe = None
    except (ValueError, TypeError):
        pass

    # PB
    pb = None
    try:
        pb = float(realtime.get("市净率", 0))
        if pb <= 0: pb = None
    except (ValueError, TypeError):
        pass

    score = 3.0
    pe_score = 3.0

    # PE 评分
    if pe is not None:
        if pe < 15: pe_score = 4.5
        elif pe < 25: pe_score = 4.0
        elif pe < 40: pe_score = 3.5
        elif pe < 60: pe_score = 3.0
        elif pe < 100: pe_score = 2.5
        elif pe < 200: pe_score = 2.0
        else: pe_score = 1.5

    # 盈利质量（从基本面数据）
    roe = None
    for k in fin:
        if "ROE" in str(k).upper() or "净资产收益率" in str(k):
            try:
                roe = float(fin[k])
                break
            except (ValueError, TypeError):
                pass

    roe_score = 3.0
    if roe is not None and roe > 0:
        if roe > 20: roe_score = 4.5
        elif roe > 15: roe_score = 4.0
        elif roe > 10: roe_score = 3.5
        elif roe > 5: roe_score = 3.0
        else: roe_score = 2.5

    score = round((pe_score * 0.6 + roe_score * 0.4), 2)

    return {
        "score": score,
        "detail": {
            "PE": round(pe, 2) if pe else None,
            "PB": round(pb, 2) if pb else None,
            "ROE": round(roe, 2) if roe else None,
        }
    }


def factor_market(df: pd.DataFrame, market: str = "cn") -> dict:
    """市场环境因子 — A股用上证，美股用标普500"""
    idx_label = "上证"
    try:
        if market == "us":
            idx = _get_sp500_index()
            if idx.empty:
                return {"score": 3.0, "detail": {"标普5日涨跌": 0, "标普20日涨跌": 0}}
            idx_label = "标普"
            idx_close = idx["close"] if "close" in idx.columns else idx["收盘"]
        else:
            idx = _get_index_daily()
            idx_close = idx["close"]
        idx_ret_5 = float((idx_close.iloc[-1] / idx_close.iloc[-6] - 1) * 100) if len(idx_close) >= 6 else 0
        idx_ret_20 = float((idx_close.iloc[-1] / idx_close.iloc[-21] - 1) * 100) if len(idx_close) >= 21 else 0
    except Exception:
        idx_ret_5, idx_ret_20 = 0, 0

    score = 3.0
    if idx_ret_5 > 2: score += 0.5
    elif idx_ret_5 < -2: score -= 0.5
    if idx_ret_20 > 5: score += 0.3
    elif idx_ret_20 < -5: score -= 0.3

    score = round(min(5.0, max(1.0, score)), 2)

    return {
        "score": score,
        "detail": {
            f"{idx_label}5日涨跌": round(idx_ret_5, 2),
            f"{idx_label}20日涨跌": round(idx_ret_20, 2),
        }
    }


# ── 股票分类 ────────────────────────────────────────────────

def classify_stock(realtime: dict, fin: dict) -> str:
    """分类：small_cap | momentum_driven | large_cap_value（mcap单位：亿）"""
    try:
        mcap = float(realtime.get("总市值", 0))
    except (ValueError, TypeError):
        mcap = 0

    try:
        pe = float(realtime.get("市盈率-动态", 0) or realtime.get("市盈率（动态）", 100))
    except (ValueError, TypeError):
        pe = 100

    if mcap > 0 and mcap < 100:  # < 100亿 → 小盘题材
        return "small_cap"
    elif pe > 80 or pe <= 0:
        return "momentum_driven"
    else:
        return "large_cap_value"


def get_weights(stock_type: str) -> dict:
    if stock_type == "small_cap":
        return {"动量": 0.35, "均值回归": 0.15, "技术指标": 0.20,
                "量价分析": 0.15, "基本面": 0.05, "市场环境": 0.10}
    elif stock_type == "momentum_driven":
        return {"动量": 0.30, "均值回归": 0.10, "技术指标": 0.20,
                "量价分析": 0.20, "基本面": 0.10, "市场环境": 0.10}
    else:  # large_cap_value
        return {"动量": 0.25, "均值回归": 0.20, "技术指标": 0.15,
                "量价分析": 0.10, "基本面": 0.20, "市场环境": 0.10}


def score_to_prob(score: float) -> float:
    """得分 1-5 → 概率 25%-75%"""
    return round(25 + (score - 1) * 12.5, 1)


def assess_risks(df: pd.DataFrame, realtime: dict, score: float) -> list[str]:
    risks = []

    try:
        t_rate = float(realtime.get("换手率", 0))
        if t_rate > 20:
            risks.append("极端换手率，存在出货或大幅波动风险")
        elif t_rate > 15:
            risks.append("换手率偏高，筹码交换剧烈")
    except Exception:
        pass

    try:
        inner = float(realtime.get("内盘", 0))
        outer = float(realtime.get("外盘", 0))
        if inner > outer * 1.5 and (inner + outer) > 0:
            risks.append("主动卖盘显著大于买盘，抛压较重")
    except Exception:
        pass

    if score < 2.5:
        risks.append("综合评分偏低，短期上涨概率不足50%")

    close = _col(df, "收盘")
    if len(close) >= 60:
        high = float(close.iloc[-60:].max())
        drawdown = (1 - float(close.iloc[-1]) / high) * 100
        if drawdown > 20:
            risks.append(f"距60日高点回撤 {drawdown:.0f}%，上方套牢盘压力大")

    return risks


# ── 核心分析 ────────────────────────────────────────────────

def analyze_stock(code: str) -> dict:
    """对单只股票运行完整多因子分析 — 自动检测A股/美股"""
    t0 = time.time()
    market = detect_market(code)

    # 获取数据
    df = _get_daily_unified(code, market, 250)
    realtime = _get_realtime_unified(code, market, df)
    fin = _get_financials_unified(code, market)

    # 美股：从财务数据补充名称
    name = str(realtime.get("名称", code))
    if market == "us" and fin.get("_display_name"):
        name = str(fin["_display_name"])
        realtime["名称"] = name

    price = float(realtime.get("最新价", 0))
    mcap = float(realtime.get("总市值", 0))
    change = float(realtime.get("涨跌幅", 0))
    pe = float(realtime.get("市盈率-动态", 0) or realtime.get("市盈率（动态）", 0) or 0)

    # 从财务数据补充 PE
    if not pe and fin:
        eps = float(fin.get("基本每股收益", 0) or fin.get("摊薄每股收益_最新股本", 0) or 0)
        if eps > 0 and price > 0:
            if market == "us":
                pe = round(price / eps, 2)  # 美股EPS通常是TTM/年报，不×4
            else:
                pe = round(price / (eps * 4), 2)
    if not mcap and fin:
        net_profit = float(fin.get("归母净利润", 0) or fin.get("净利润", 0) or 0)
        if net_profit > 0 and pe > 0:
            if market == "us":
                mcap = round(pe * net_profit / 1e8, 2)  # 美元→折合亿
            else:
                mcap = round(pe * net_profit * 4 / 1e8, 2)
    realtime["市盈率-动态"] = pe
    realtime["总市值"] = mcap

    # 分类 + 权重（美股大市值阈值提高到500亿）
    if market == "us" and mcap > 500:
        stock_type = "large_cap_value"
    elif market == "us":
        stock_type = "momentum_driven"
    else:
        stock_type = classify_stock(realtime, fin)
    weights = get_weights(stock_type)

    # 计算各因子
    factors = {
        "动量": factor_momentum(df, realtime),
        "均值回归": factor_reversion(df),
        "技术指标": factor_technical(df),
        "量价分析": factor_volume(df, realtime),
        "基本面": factor_fundamental(fin, realtime, df),
        "市场环境": factor_market(df, market),
    }

    # 加权汇总
    total = sum(factors[k]["score"] * weights.get(k, 0.1) for k in factors)
    total = round(total, 2)
    prob = score_to_prob(total)

    # 风险
    risks = assess_risks(df, realtime, total)

    # 均线
    close = _col(df, "收盘")
    ma_list = {}
    for m in [5, 10, 20, 30, 60, 120, 250]:
        if len(close) >= m:
            ma_list[f"MA{m}"] = round(float(calc_ma(close, m).iloc[-1]), 2)

    # 各周期涨跌幅
    rets = {}
    for label, days in [("5日", 5), ("10日", 10), ("20日", 20), ("60日", 60)]:
        if len(close) > days:
            rets[label] = round((close.iloc[-1] / close.iloc[-(days+1)] - 1) * 100, 2)

    elapsed = round((time.time() - t0) * 1000)

    return {
        "code": code,
        "market": market,
        "name": name,
        "price": round(price, 2),
        "change_pct": round(change, 2),
        "market_cap": round(mcap, 1) if mcap > 0 else None,
        "pe": round(pe, 2) if pe > 0 else None,
        "stock_type": stock_type,
        "weights": {k: round(v, 2) for k, v in weights.items()},
        "factors": {
            k: {"score": v["score"], "detail": v["detail"]}
            for k, v in factors.items()
        },
        "total_score": total,
        "probability": prob,
        "risks": risks,
        "ma_values": ma_list,
        "returns": rets,
        "elapsed_ms": elapsed,
    }


def compare_stocks(queries: list[str]) -> dict:
    """多票横向对比 — 自动检测每只的市场"""
    results = []
    for q in queries:
        try:
            r = analyze_stock(q.strip())
            results.append(r)
        except Exception as e:
            results.append({"code": q.strip(), "error": str(e)})

    return {
        "count": len(results),
        "comparison": [
            {
                "code": r.get("code"),
                "market": r.get("market", "cn"),
                "name": r.get("name", "?"),
                "price": r.get("price"),
                "change": r.get("change_pct"),
                "pe": r.get("pe"),
                "type": r.get("stock_type"),
                "score": r.get("total_score"),
                "probability": r.get("probability"),
            }
            for r in results if "error" not in r
        ],
        "errors": [r for r in results if "error" in r],
        "results": results,
    }


# ── API 路由 ────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.get("/api/search")
async def api_search(q: str = Query(..., min_length=1)):
    results = search_stock(q)
    return {"query": q, "count": len(results), "results": results}


@app.get("/api/analyze")
async def api_analyze(code: str = Query(..., min_length=1)):
    try:
        result = analyze_stock(code.strip())
        return JSONResponse(result)
    except ValueError as e:
        return JSONResponse({"error": str(e), "code": code}, status_code=400)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb, flush=True)
        return JSONResponse({"error": str(e), "code": code, "traceback": tb}, status_code=500)


@app.get("/api/compare")
async def api_compare(codes: str = Query(..., min_length=1)):
    code_list = [c.strip() for c in codes.split(",") if c.strip()]
    if len(code_list) < 2:
        return JSONResponse({"error": "至少需要2个股票代码"}, status_code=400)
    try:
        result = compare_stocks(code_list)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── 前端页面 ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── 启动入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
