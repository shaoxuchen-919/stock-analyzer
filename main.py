"""
=============================================================================
  A股+美股 多因子量化分析系统  —  FastAPI + yfinance
  单票分析 + 多票对比 + 动态权重 + 概率评估
  数据源: Yahoo Finance (全球可访问)
=============================================================================
"""

from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="A股+美股多因子量化分析", version="3.0.0")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 缓存 ───────────────────────────────────────────────────
_cache: dict[str, tuple[float, Any]] = {}
CACHE_TTL = 900  # 15分钟（Yahoo数据更新频率）


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


# ── 市场检测 & Yahoo Ticker 转换 ───────────────────────────

def detect_market(code: str) -> str:
    """检测市场: 'cn' = A股, 'us' = 美股"""
    code = code.strip().upper()
    if re.match(r'^\d{6}$', code):
        return 'cn'
    if re.match(r'^[A-Z]{1,5}$', code):
        return 'us'
    raise ValueError(f"无法识别市场: {code}，请输入6位A股代码或美股ticker（如 AAPL）")


def _cn_code_to_yahoo(code: str) -> str:
    """6位A股代码 → Yahoo Finance ticker"""
    code = code.strip().upper()
    if code.startswith('6'):
        return f"{code}.SS"   # 上海
    elif code.startswith(('0', '3')):
        return f"{code}.SZ"   # 深圳
    elif code.startswith(('4', '8')):
        return f"{code}.BJ"   # 北京
    else:
        return f"{code}.SZ"   # 默认深圳


def _yf_download(ticker: str, period: str = "1y") -> pd.DataFrame:
    """统一的 yfinance 下载接口，带错误处理"""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError(f"Yahoo Finance 返回空数据: {ticker}")
        # 标准化列名
        df = df.reset_index()
        # yfinance 可能返回 Date 或 Datetime 列
        date_col = None
        for c in df.columns:
            if 'date' in str(c).lower() or 'datetime' in str(c).lower():
                date_col = c
                break
        if date_col:
            df.rename(columns={date_col: "日期"}, inplace=True)
        # 标准化价格列名
        col_map = {"Open": "开盘", "High": "最高", "Low": "最低",
                    "Close": "收盘", "Volume": "成交量"}
        df = df.rename(columns=col_map)
        # 去掉收盘价为 NaN 的行（当日无数据）
        if "收盘" in df.columns:
            df = df.dropna(subset=["收盘"])
        return df
    except Exception as e:
        raise ValueError(f"获取 {ticker} 数据失败: {e}")


# ── 数据获取（统一用 yfinance）─────────────────────────────

def _fetch_daily(code: str, market: str, days: int = 250) -> pd.DataFrame:
    """获取日K线 — A股和美股统一走 Yahoo Finance"""
    if market == 'cn':
        ticker = _cn_code_to_yahoo(code)
    else:
        ticker = code

    # 多抓一些数据确保够 days 天
    period_map = {60: "3mo", 120: "6mo", 250: "1y", 500: "2y"}
    period = period_map.get(max(days * 2, 250), "2y")

    df = _yf_download(ticker, period)

    if len(df) < 5:
        raise ValueError(f"数据量不足（仅{len(df)}天）: {code}")

    # 补充计算列
    df["涨跌幅"] = df["收盘"].pct_change() * 100
    df["涨跌额"] = df["收盘"].diff()
    if "成交额" not in df.columns:
        df["成交额"] = df["成交量"] * df["收盘"]  # 估算
    if "换手率" not in df.columns:
        df["换手率"] = 0.0  # Yahoo 不提供换手率

    df = df.tail(days).reset_index(drop=True)
    return df


def _get_daily_unified(code: str, market: str, days: int = 250) -> pd.DataFrame:
    key = _cache_key("daily", code)
    cached = _cache_get(key)
    if cached is not None and len(cached) >= days:
        return cached.tail(days).reset_index(drop=True)
    df = _fetch_daily(code, market, days)
    _cache_set(key, df)
    return df.tail(days).reset_index(drop=True)


def _get_realtime_unified(code: str, market: str, daily_df: pd.DataFrame) -> dict:
    """从日K线提取实时行情"""
    row = daily_df.iloc[-1]
    price = float(row["收盘"])
    prev_price = float(daily_df["收盘"].iloc[-2]) if len(daily_df) >= 2 else price
    change_pct = round((price / prev_price - 1) * 100, 2) if prev_price > 0 else 0

    result = {
        "代码": code,
        "名称": code,
        "最新价": price,
        "涨跌幅": change_pct,
        "换手率": float(row.get("换手率", 0)),
        "总市值": 0,       # 由财务数据补充
        "市盈率-动态": 0,   # 由财务数据补充
        "成交量": float(row.get("成交量", 0)),
        "成交额": float(row.get("成交额", 0)),
        "内盘": 0,
        "外盘": 0,
    }

    # 尝试从 yfinance Ticker 获取更多实时信息
    try:
        if market == 'cn':
            ticker = _cn_code_to_yahoo(code)
        else:
            ticker = code
        t = yf.Ticker(ticker)
        info = t.info
        if info:
            # 名称
            if info.get("shortName"):
                result["名称"] = info["shortName"]
            elif info.get("longName"):
                result["名称"] = info["longName"]
            # 市值 (Yahoo 返回的是原始单位，需转换)
            mcap_raw = info.get("marketCap")
            if mcap_raw:
                if market == 'us':
                    result["总市值"] = round(mcap_raw / 1e8, 2)  # USD → 亿
                else:
                    result["总市值"] = round(mcap_raw / 1e8, 2)  # CNY → 亿
            # PE
            pe_raw = info.get("trailingPE") or info.get("forwardPE")
            if pe_raw:
                result["市盈率-动态"] = round(pe_raw, 2)
            # PB
            pb_raw = info.get("priceToBook")
            if pb_raw:
                result["市净率"] = round(pb_raw, 2)
            # 52周高低
            result["52周最高"] = info.get("fiftyTwoWeekHigh")
            result["52周最低"] = info.get("fiftyTwoWeekLow")
    except Exception:
        pass

    return result


def _get_financials_unified(code: str, market: str) -> dict:
    """获取基本面数据 — 统一走 yfinance"""
    key = _cache_key("fin", code)
    cached = _cache_get(key)
    if cached is not None:
        records = cached.to_dict("records")
        if records:
            return records[0]
        return {}

    try:
        if market == 'cn':
            ticker = _cn_code_to_yahoo(code)
        else:
            ticker = code

        t = yf.Ticker(ticker)
        info = t.info or {}
        result = {}

        # 提取关键财务指标
        field_map = {
            "trailingEps": "基本每股收益",
            "forwardEps": "预测每股收益",
            "returnOnEquity": "净资产收益率",
            "returnOnAssets": "资产收益率",
            "grossMargins": "毛利率",
            "operatingMargins": "营业利润率",
            "profitMargins": "净利润率",
            "revenueGrowth": "营收增长率",
            "earningsGrowth": "盈利增长率",
            "totalRevenue": "营业总收入",
            "netIncomeToCommon": "归母净利润",
            "totalDebt": "总负债",
            "totalCash": "总现金",
            "bookValue": "每股净资产",
            "priceToBook": "市净率",
            "trailingPE": "市盈率(TTM)",
            "forwardPE": "市盈率(前瞻)",
            "pegRatio": "PEG",
            "dividendYield": "股息率",
            "payoutRatio": "分红比率",
            "beta": "Beta",
            "enterpriseValue": "企业价值",
            "ebitda": "EBITDA",
            "operatingCashflow": "经营现金流",
            "freeCashflow": "自由现金流",
            "sharesOutstanding": "总股本",
        }

        for en, cn in field_map.items():
            val = info.get(en)
            if val is not None and pd.notna(val):
                try:
                    result[cn] = float(val)
                except (ValueError, TypeError):
                    result[cn] = val

        # 显示名称
        name = info.get("shortName") or info.get("longName")
        if name:
            result["_display_name"] = str(name)

        _cache_set(key, pd.DataFrame([result]))
        return result
    except Exception as e:
        pass

    empty = {}
    _cache_set(key, pd.DataFrame([empty]))
    return empty


# ── 指数数据（Yahoo Finance）────────────────────────────────

def _get_index_daily() -> pd.DataFrame:
    """获取上证指数日K线 (000001.SS)"""
    key = _cache_key("idx", "sh000001")
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        df = _yf_download("000001.SS", "6mo")
        df = df.tail(120).reset_index(drop=True)
        _cache_set(key, df)
        return df
    except Exception:
        return pd.DataFrame()


def _get_sp500_index() -> pd.DataFrame:
    """获取标普500日K线 (^GSPC)"""
    key = _cache_key("idx", "sp500")
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        df = _yf_download("^GSPC", "6mo")
        df = df.tail(120).reset_index(drop=True)
        _cache_set(key, df)
        return df
    except Exception:
        return pd.DataFrame()


# ── 搜索 ────────────────────────────────────────────────────

# ── 搜索（纯本地映射 + yfinance 验证）──────────────────────

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

# 常用 A 股映射（热门股）
_CN_POPULAR: dict[str, str] = {
    "600519": "贵州茅台", "601318": "中国平安", "600036": "招商银行",
    "601398": "工商银行", "601988": "中国银行", "601288": "农业银行",
    "600276": "恒瑞医药", "002594": "比亚迪", "300750": "宁德时代",
    "600900": "长江电力", "601012": "隆基绿能", "002475": "立讯精密",
    "600584": "长电科技", "002962": "五方光电", "300390": "天华新能",
    "000858": "五粮液", "000333": "美的集团", "601888": "中国中车",
    "002230": "科大讯飞", "688981": "中芯国际", "002371": "北方华创",
    "300059": "东方财富", "601166": "兴业银行", "600809": "山西汾酒",
    "000651": "格力电器", "002714": "牧原股份", "603259": "药明康德",
    "300122": "智飞生物", "002304": "洋河股份", "600887": "伊利股份",
    "000568": "泸州老窖", "603288": "海天味业", "002352": "顺丰控股",
    "601899": "紫金矿业", "601985": "中国核电", "00381": "中国石油",
    "600028": "中国石化", "601088": "中国神华", "601668": "中国建筑",
}

# 英文别名 → ticker
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


def search_stock(name: str, limit: int = 10) -> list[dict]:
    """搜索股票 — 纯本地匹配，无外部API调用"""
    name_stripped = name.strip()
    name_upper = name_stripped.upper()
    results = []

    # 1. 6位数字 → A股精确匹配
    if re.match(r'^\d{6}$', name_upper):
        cn_name = _CN_POPULAR.get(name_upper, f"A股{name_upper}")
        results.append({"代码": name_upper, "名称": cn_name, "市场": "A股"})
        return results[:limit]

    # 2. 美股 ticker 精确/前缀匹配
    if re.match(r'^[A-Z]{1,5}$', name_upper):
        if name_upper in _US_POPULAR:
            results.append({"代码": name_upper, "名称": _US_POPULAR[name_upper], "市场": "美股"})
        for ticker, cname in _US_POPULAR.items():
            if ticker.startswith(name_upper) and ticker != name_upper:
                results.append({"代码": ticker, "名称": cname, "市场": "美股"})
            if len(results) >= limit:
                return results[:limit]
        # 英文别名
        ticker_from_alias = _US_ALIAS.get(name_upper)
        if ticker_from_alias and ticker_from_alias in _US_POPULAR:
            entry = {"代码": ticker_from_alias, "名称": _US_POPULAR[ticker_from_alias], "市场": "美股"}
            if entry not in results:
                results.append(entry)
        return results[:limit]

    # 3. 中文名模糊搜索 — 美股
    for ticker, cname in _US_POPULAR.items():
        if name_stripped in cname or name_upper in ticker.upper():
            entry = {"代码": ticker, "名称": cname, "市场": "美股"}
            if entry not in results:
                results.append(entry)
        if len(results) >= limit:
            return results[:limit]

    # 4. 英文别名匹配
    for alias, ticker in _US_ALIAS.items():
        if name_upper in alias or alias.startswith(name_upper):
            ticker_cn = _US_POPULAR.get(ticker, ticker)
            entry = {"代码": ticker, "名称": ticker_cn, "市场": "美股"}
            if entry not in results:
                results.append(entry)
        if len(results) >= limit:
            return results[:limit]

    # 5. 中文名搜索 — A股
    looks_chinese = bool(re.search(r'[\u4e00-\u9fff]', name))
    if len(results) < limit and looks_chinese:
        for code, cname in _CN_POPULAR.items():
            if name_stripped in cname:
                results.append({"代码": code, "名称": cname, "市场": "A股"})
            if len(results) >= limit:
                return results[:limit]

    return results[:limit]


@lru_cache(maxsize=1)
def _get_stock_list_df() -> pd.DataFrame:
    """兼容旧接口，返回空DataFrame（已切换到本地映射）"""
    return pd.DataFrame()


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
        else:
            idx = _get_index_daily()
        # 统一取收盘价列
        if "收盘" in idx.columns:
            idx_close = idx["收盘"]
        elif "close" in idx.columns:
            idx_close = idx["close"]
        else:
            raise ValueError("无收盘价列")
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
async def index():
    index_path = os.path.join(BASE_DIR, "templates", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── 启动入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
