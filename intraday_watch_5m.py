import os
import json
import math
import time
from datetime import datetime, timedelta
from dateutil import tz

import requests
import pandas as pd
import akshare as ak


BJ = tz.gettz("Asia/Shanghai")
STATE_PATH = "intraday_state.json"

# 固定：TopN=5
TOPN = 5

# 事件优先级（越小越重要）
PRIO = {
    # 风险类（最影响决策）
    "VOL_DOWN": 1,
    "VOL_STALL": 1,

    # 锚位突破/失守（跨日锚优先）
    "ANCHOR_LOSE_PREV_H": 2,
    "ANCHOR_LOSE_PREV_C": 2,
    "ANCHOR_LOSE_PREV_L": 2,
    "ANCHOR_BREAK_PREV_H": 2,
    "ANCHOR_BREAK_PREV_C": 2,
    "ANCHOR_BREAK_PREV_L": 2,

    # ORB
    "ORB_UP_BREAK": 3,
    "ORB_DN_BREAK": 3,

    # 放量确认/Spike（确认）
    "BREAK_CONFIRM_VOL": 4,
    "AMT_SPIKE_5M": 4,
    "AMT_SPIKE_1M": 4,

    # proxy（⑤⑥⑦）
    "ACTIVE_PROXY_SHIFT": 6,
    "LIMIT_TOUCH": 6,
    "LIMIT_SEALED": 6,
    "LIMIT_BREAK": 6,
    "LIMIT_RESEAL": 6,
}


# =========================
# Time helpers
# =========================
def now_bj() -> datetime:
    return datetime.now(tz=BJ)

def fmt_dt(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")

def in_trading_session(t: datetime) -> bool:
    hm = t.hour * 60 + t.minute
    am = (9 * 60 + 30, 11 * 60 + 30)
    pm = (13 * 60, 15 * 60)
    return (am[0] <= hm <= am[1]) or (pm[0] <= hm <= pm[1])

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def pct_change(a, b):
    if a is None or b in (None, 0):
        return None
    return (a / b - 1.0) * 100.0

def ma(arr, n):
    if arr is None or len(arr) < n:
        return None
    return sum(arr[-n:]) / n

def std(arr, n):
    if arr is None or len(arr) < n:
        return None
    m = ma(arr, n)
    v = sum((x - m) ** 2 for x in arr[-n:]) / n
    return math.sqrt(v)

def ma_dyn(arr, n, min_len=8):
    """动态窗口均值：min(n, len(arr))，但至少 min_len 个样本才算。"""
    if arr is None:
        return None
    k = min(n, len(arr))
    if k < min_len:
        return None
    return sum(arr[-k:]) / k

def retry(fn, times=3, sleep_s=1.2, desc=""):
    last = None
    for i in range(times):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(sleep_s * (i + 1))
    raise last


# =========================
# State (cooldown / ORB)
# =========================
def load_state():
    if not os.path.exists(STATE_PATH):
        return {"date": "", "orb": {}, "last_push": {}}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"date": "", "orb": {}, "last_push": {}}

def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def should_push(state, key: str, cooldown_min: int, now_ts: float) -> bool:
    last = state.get("last_push", {}).get(key)
    if last is None:
        return True
    try:
        return (now_ts - float(last)) >= cooldown_min * 60
    except Exception:
        return True

def mark_push(state, key: str, now_ts: float):
    state.setdefault("last_push", {})[key] = now_ts


# =========================
# Feishu push
# =========================
def feishu_send_text(webhook: str, text: str):
    payload = {"msg_type": "text", "content": {"text": text}}
    r = requests.post(webhook, json=payload, timeout=15)
    r.raise_for_status()


# =========================
# LLM (OpenAI-compatible)
# =========================
def llm_analyze(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content":
                "你是A股盘中监控与交易分析助手。"
                "必须输出可复核结论：每条结论必须包含“日期时间+数值”证据。"
                "⑤⑥⑦属于proxy近似，必须明确标注不可替代L2逐笔/盘口。"
                "输出结构固定："
                "1) 一句话结论(<=30字)"
                "2) 证据表(>=5条：日期时间+数值+对应事件)"
                "3) 推理链(基于证据逐条推导，禁止空话)"
                "4) 行动建议(观察/入场/减仓/离场 + 触发条件/无效条件/风控线，必须给数值)"
            },
            {"role": "user", "content": prompt},
        ],
    }
    r = requests.post(url, headers=headers, json=body, timeout=35)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# =========================
# Time parser (Must-fix 1.3)
# =========================
def _parse_time_any(x):
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=BJ)
    if isinstance(x, pd.Timestamp):
        dt = x.to_pydatetime()
        return dt if dt.tzinfo else dt.replace(tzinfo=BJ)

    s = str(x).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=BJ)
        except Exception:
            pass

    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=BJ)
    except Exception:
        raise ValueError(f"Unrecognized time format: {repr(x)}")


# =========================
# AKShare fetch + normalize
# =========================
def normalize_min_df(df):
    """
    标准化成列：time, open, high, low, close, vol, amt
    Must: 输入为空直接 raise（不要返回 None）
    """
    if df is None or df.empty:
        raise ValueError("normalize_min_df: input df is None or empty")

    cols = set(df.columns)
    time_col = "时间" if "时间" in cols else ("日期" if "日期" in cols else None)
    open_col = "开盘" if "开盘" in cols else None
    high_col = "最高" if "最高" in cols else None
    low_col  = "最低" if "最低" in cols else None
    close_col= "收盘" if "收盘" in cols else ("最新价" if "最新价" in cols else None)
    vol_col  = "成交量" if "成交量" in cols else None
    amt_col  = "成交额" if "成交额" in cols else None

    if not time_col or not open_col or not high_col or not low_col or not close_col:
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    out = df[[time_col, open_col, high_col, low_col, close_col]].copy()
    out.columns = ["time", "open", "high", "low", "close"]
    out["vol"] = df[vol_col].apply(safe_float) if vol_col else None
    out["amt"] = df[amt_col].apply(safe_float) if amt_col else None

    out["time"] = out["time"].apply(_parse_time_any)
    return out.sort_values("time").reset_index(drop=True)

def fetch_stock_min(symbol: str, period: str, start_dt: str, end_dt: str):
    def _fn():
        return ak.stock_zh_a_hist_min_em(
            symbol=symbol, period=period, start_date=start_dt, end_date=end_dt, adjust=""
        )
    return retry(_fn, desc=f"stock_min {symbol} {period}")

def fetch_index_min(symbol: str, period: str, start_dt: str, end_dt: str):
    def _fn():
        return ak.index_zh_a_hist_min_em(
            symbol=symbol, period=period, start_date=start_dt, end_date=end_dt
        )
    return retry(_fn, desc=f"index_min {symbol} {period}")


# =========================
# Enhancement: 1m -> 5m resample (fallback)
# =========================
def resample_1m_to_5m(df1_norm):
    """
    用 1m 聚合成 5m（保证 5m 源断连时不断档）
    输出列：time/open/high/low/close/vol/amt
    """
    if df1_norm is None or df1_norm.empty:
        return None

    d = df1_norm.copy()
    d["bucket"] = d["time"].dt.floor("5min")

    agg = d.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        vol=("vol", "sum"),
        amt=("amt", "sum"),
    ).reset_index().rename(columns={"bucket": "time"})

    # 丢掉尚未完成的最后一桶（防止当前5m桶未收盘就聚合）
    nowt = now_bj()
    if not agg.empty:
        last_bucket = agg["time"].iloc[-1]
        if (last_bucket + timedelta(minutes=5)) > nowt:
            agg = agg.iloc[:-1]

    return agg.reset_index(drop=True)


# =========================
# Must-fix 1.1: prev OHLC date selection
# =========================
def fetch_prev_day_ohlc(symbol: str, yyyymmdd: str):
    """
    修复逻辑：
      - 如果日线最后一行日期 == 今天：prev = -2
      - 否则：prev = -1
    """
    end_date = datetime.strptime(yyyymmdd, "%Y%m%d").date()
    start_date = (end_date - timedelta(days=60)).strftime("%Y%m%d")

    def _fn():
        return ak.stock_zh_a_hist(
            symbol=symbol, period="daily", start_date=start_date, end_date=yyyymmdd, adjust=""
        )

    df = retry(_fn, desc=f"daily {symbol}")
    if df is None or df.empty or "日期" not in df.columns:
        return None

    df = df.sort_values("日期").reset_index(drop=True)
    last_date = pd.to_datetime(df.iloc[-1]["日期"]).date()

    if last_date == end_date:
        if len(df) < 2:
            return None
        prev = df.iloc[-2]
    else:
        prev = df.iloc[-1]

    return {
        "date": str(prev["日期"]),
        "open": safe_float(prev.get("开盘")),
        "high": safe_float(prev.get("最高")),
        "low": safe_float(prev.get("最低")),
        "close": safe_float(prev.get("收盘")),
    }


# =========================
# Bars / ORB
# =========================
def get_last_completed_bar(df_norm, cutoff: datetime):
    if df_norm is None or getattr(df_norm, "empty", True):
        return None
    d2 = df_norm[df_norm["time"] <= cutoff]
    if d2.empty:
        return None
    return d2.iloc[-1]

def calc_orb(one_min_norm, orb_buffer_bp: int):
    """
    ORB: 09:30-09:34
    同时返回 ORB 1m 平均成交额，作为早盘 Spike 的兜底基准
    """
    if one_min_norm is None or one_min_norm.empty:
        return None
    day = one_min_norm["time"].iloc[0].date()
    s = datetime(day.year, day.month, day.day, 9, 30, tzinfo=BJ)
    e = datetime(day.year, day.month, day.day, 9, 35, tzinfo=BJ)

    w = one_min_norm[(one_min_norm["time"] >= s) & (one_min_norm["time"] < e)]
    if w.empty or len(w) < 3:
        return None

    orb_h = float(w["high"].max())
    orb_l = float(w["low"].min())
    buf = orb_buffer_bp / 10000.0

    amt_series = w["amt"].fillna(0).astype(float).tolist()
    orb_amt_mean_1m = float(sum(amt_series) / max(len(amt_series), 1))

    return {"orb_h": orb_h, "orb_l": orb_l, "buf": buf, "orb_amt_mean_1m": orb_amt_mean_1m}

def estimate_limit_up(symbol: str, prev_close: float):
    # proxy：300/688 20%，其他 10%（未处理ST等特殊涨跌幅）
    pct = 0.2 if (symbol.startswith("300") or symbol.startswith("688")) else 0.1
    return round(prev_close * (1 + pct), 2)

def limitup_proxy(one_min_norm, limit_up: float):
    tol = 0.01
    df = one_min_norm
    touched = df[df["high"] >= limit_up - tol]
    if touched.empty:
        return None
    first_touch = touched["time"].iloc[0]

    sealed = df[df["close"] >= limit_up - tol]
    if sealed.empty:
        return {"type": "LIMIT_TOUCH", "dir": "up",
                "facts": {"t_first_touch": fmt_dt(first_touch), "limit_up": limit_up, "note": "⑥⑦ proxy：分钟级触板"}}

    first_seal = sealed["time"].iloc[0]
    after = df[df["time"] >= first_seal]
    opened = after[after["close"] < limit_up - tol]

    if opened.empty:
        seal_mins = int((after["close"] >= limit_up - tol).sum())
        return {"type": "LIMIT_SEALED", "dir": "up",
                "facts": {"t_first_seal": fmt_dt(first_seal), "limit_up": limit_up,
                          "seal_minutes_proxy": seal_mins, "note": "⑥⑦ proxy：封板持续分钟数(非封单金额/撤单率)"}}

    first_open = opened["time"].iloc[0]
    after_open = after[after["time"] >= first_open]
    reseal = after_open[after_open["close"] >= limit_up - tol]

    if reseal.empty:
        return {"type": "LIMIT_BREAK", "dir": "down",
                "facts": {"t_first_seal": fmt_dt(first_seal), "t_first_open": fmt_dt(first_open),
                          "limit_up": limit_up, "note": "⑥⑦ proxy：炸板后未回封(分钟级)"}}

    reseal_t = reseal["time"].iloc[0]
    minutes = int((reseal_t - first_open).total_seconds() / 60)
    return {"type": "LIMIT_RESEAL", "dir": "up",
            "facts": {"t_first_open": fmt_dt(first_open), "t_reseal": fmt_dt(reseal_t),
                      "reseal_minutes_proxy": minutes, "limit_up": limit_up,
                      "note": "⑥⑦ proxy：回封耗时(分钟级)"}}


# =========================
# Should-fix 2.3: active proxy with guards
# =========================
def active_proxy(one_min_norm, min_tot_amt_last30=5e6):
    """
    ⑤ proxy：上涨K线成交额占比（近30m vs 前60m）
    防误判：要求近30m总成交额>阈值且不为0
    """
    df = one_min_norm.copy()
    cutoff = now_bj() - timedelta(minutes=1)
    df = df[df["time"] <= cutoff]
    if len(df) < 120:
        return None

    last30 = df.iloc[-30:]
    prev60 = df.iloc[-90:-30]

    tot1 = float(last30["amt"].fillna(0).sum())
    tot0 = float(prev60["amt"].fillna(0).sum())
    if tot1 <= 0 or tot0 <= 0 or tot1 < min_tot_amt_last30:
        return None

    up1 = float(last30[last30["close"] > last30["open"]]["amt"].fillna(0).sum())
    up0 = float(prev60[prev60["close"] > prev60["open"]]["amt"].fillna(0).sum())

    r1 = up1 / tot1
    r0 = up0 / tot0
    delta = r1 - r0

    return {"t": fmt_dt(df["time"].iloc[-1]),
            "up_amt_ratio_last30": r1, "up_amt_ratio_prev60": r0, "delta": delta,
            "tot_amt_last30": tot1,
            "note": "⑤ proxy：以上涨K线成交额占比近似主动买盘占比(非逐笔主动买)"}  # 明确边界


# =========================
# Must-fix 1.2: RS aligned by time join
# =========================
def calc_rs_z_aligned(stock_5m_norm, index_5m_norm, cutoff: datetime, window: int = 20):
    """
    ⑧ 正确性版本：按 time inner join 对齐后计算 rs_series rolling mean/std
    rs = (stock_close/stock_open-1) - (idx_close/idx_open-1)
    """
    if stock_5m_norm is None or stock_5m_norm.empty:
        return None
    if index_5m_norm is None or index_5m_norm.empty:
        return None

    s = stock_5m_norm[stock_5m_norm["time"] <= cutoff][["time","open","close"]].copy()
    i = index_5m_norm[index_5m_norm["time"] <= cutoff][["time","open","close"]].copy()
    if s.empty or i.empty:
        return None

    m = s.merge(i, on="time", how="inner", suffixes=("_s","_i")).sort_values("time")
    if len(m) < window + 5:
        return None

    # 避免除0
    m = m[(m["open_s"] != 0) & (m["open_i"] != 0)]
    if len(m) < window + 5:
        return None

    m["ret_s"] = m["close_s"] / m["open_s"] - 1.0
    m["ret_i"] = m["close_i"] / m["open_i"] - 1.0
    m["rs"] = m["ret_s"] - m["ret_i"]

    rs_series = m["rs"].astype(float).tolist()
    rs_last = float(rs_series[-1])
    mu = ma(rs_series, window)
    sd = std(rs_series, window)
    if mu is None or sd in (None, 0):
        return None

    z = float((rs_last - mu) / sd)
    last_t = m["time"].iloc[-1]
    return {"t": fmt_dt(last_t), "rs": rs_last, "z": z}


# =========================
# Event ranking / prompt
# =========================
def event_rank(e):
    tp = e["type"]
    if tp.startswith("RS_DIVERGENCE_"):
        return 5
    return PRIO.get(tp, 99)

def build_prompt(symbol: str, idx_list: str, send_events: list, rest_events: list, context: dict):
    lines = []
    lines.append(f"标的: {symbol}")
    lines.append(f"北京时间: {context['now_bj']}")
    lines.append(f"基准指数: {idx_list}")
    lines.append("")
    lines.append(f"触发事件（TopN={TOPN}，证据必须带日期时间+数值）：")
    for e in send_events:
        lines.append(f"- {e['type']} | dir={e['dir']} | facts={json.dumps(e['facts'], ensure_ascii=False)}")
    if rest_events:
        lines.append("")
        lines.append("附录(未入LLM，仅作提示): " + ", ".join([x["type"] for x in rest_events[:15]]))
    lines.append("")
    lines.append("注意：⑤⑥⑦为proxy近似，必须显式标注，不可当作L2逐笔/盘口结论。")
    return "\n".join(lines)


# =========================
# Main
# =========================
def main():
    # -------- env ----------
    webhook = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    stock_list = os.getenv("STOCK_LIST", "").strip()
    index_list = os.getenv("INDEX_LIST", "000300,000001").strip()

    cooldown_min = int(os.getenv("COOLDOWN_MIN", "15"))
    orb_buffer_bp = int(os.getenv("ORB_BUFFER_BP", "10"))
    max_tokens = int(os.getenv("MAX_TOKENS", "450"))

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "deepseek-chat").strip()

    force_run = os.getenv("FORCE_RUN", "0").strip() == "1"

    if not webhook:
        raise SystemExit("Missing FEISHU_WEBHOOK_URL")
    if not stock_list:
        raise SystemExit("Missing STOCK_LIST")
    if not (base_url and api_key and model):
        raise SystemExit("Missing OPENAI_BASE_URL/OPENAI_API_KEY/OPENAI_MODEL")

    symbols = [s.strip() for s in stock_list.split(",") if s.strip()]
    idx_codes = [x.strip() for x in index_list.split(",") if x.strip()]

    t = now_bj()
    if (not force_run) and (not in_trading_session(t)):
        print(f"Not in trading session: {t.isoformat()}; exit.")
        return

    today = t.strftime("%Y-%m-%d")
    yyyymmdd = t.strftime("%Y%m%d")
    start_dt = f"{today} 09:30:00"
    end_dt = f"{today} 15:00:00"

    # workflow建议 sleep 15，这里仍用 now-1m 取“已完成bar”
    cutoff = t - timedelta(minutes=1)

    state = load_state()
    if state.get("date") != today:
        state = {"date": today, "orb": {}, "last_push": {}}

    now_ts = time.time()

    # 错误推送冷却（统一入口）
    def push_err(sym: str, err_key: str, text: str):
        k = f"{sym}|{err_key}"
        if should_push(state, k, cooldown_min, now_ts):
            feishu_send_text(webhook, text)
            mark_push(state, k, now_ts)

    # -------- index 5m (with fallback 1m->5m) ----------
    idx_5m = {}
    for idx in idx_codes:
        idxdf = None
        # 先取 5m
        try:
            raw = fetch_index_min(idx, "5", start_dt, end_dt)
            idxdf = normalize_min_df(raw)
        except Exception as e:
            idxdf = None
            print(f"Index 5m fetch failed {idx}: {repr(e)}")

        # fallback：1m->5m
        if idxdf is None or idxdf.empty:
            try:
                raw1 = fetch_index_min(idx, "1", start_dt, end_dt)
                idx1 = normalize_min_df(raw1)
                idxdf = resample_1m_to_5m(idx1)
                if idxdf is None or idxdf.empty:
                    idxdf = None
                else:
                    print(f"Index fallback ok {idx}: 1m->5m")
            except Exception as e:
                print(f"Index 1m fallback failed {idx}: {repr(e)}")
                idxdf = None

        idx_5m[idx] = idxdf

    # -------- per stock ----------
    for sym in symbols:
        # 1) 拉 1m（关键，可聚合5m）
        df1 = None
        try:
            raw_1m = fetch_stock_min(sym, "1", start_dt, end_dt)
            df1 = normalize_min_df(raw_1m)
        except Exception as e:
            df1 = None
            push_err(sym, "FETCH_1M_FAIL", f"【盘中5m监控】{today} {sym}\n1m数据拉取失败: {repr(e)}")

        # 2) 拉 5m
        df5 = None
        try:
            raw_5m = fetch_stock_min(sym, "5", start_dt, end_dt)
            df5 = normalize_min_df(raw_5m)
        except Exception as e:
            df5 = None
            print(f"5m fetch failed {sym}: {repr(e)}")

        # 3) 5m fallback：1m聚合5m
        if df5 is None or df5.empty:
            if df1 is not None and (not df1.empty):
                df5 = resample_1m_to_5m(df1)
                if df5 is None or getattr(df5, "empty", True):
                    push_err(sym, "RESAMPLE_5M_FAIL", f"【盘中5m监控】{today} {sym}\n1m聚合5m失败或为空，本轮跳过")
                    continue
                else:
                    push_err(sym, "DEGRADE_5M", f"【盘中5m监控】{today} {sym}\n5m源不可用，已用1m聚合生成5m（降级运行）")
            else:
                push_err(sym, "FETCH_BOTH_EMPTY", f"【盘中5m监控】{today} {sym}\n1m/5m均不可用，本轮跳过（监控不断档）")
                continue

        one_min_available = (df1 is not None and (not df1.empty))

        # 取最后已完成的 5m bar
        last5 = get_last_completed_bar(df5, cutoff)
        if last5 is None:
            # 数据尚未更新到上一根完成bar，跳过本轮
            continue

        # 昨日 OHLC（Must-fix 1.1）
        prev = None
        try:
            prev = fetch_prev_day_ohlc(sym, yyyymmdd)
        except Exception as e:
            print(f"Prev day fetch failed {sym}: {repr(e)}")

        # ORB（如果1m可用则计算；否则尝试用state里同日缓存）
        orb = None
        if one_min_available:
            try:
                orb = calc_orb(df1, orb_buffer_bp)
                if orb:
                    state.setdefault("orb", {})[sym] = orb
            except Exception as e:
                print(f"ORB calc failed {sym}: {repr(e)}")
        orb = state.get("orb", {}).get(sym)

        # 当前 5m bar 数据
        bar_t = last5["time"]
        o = safe_float(last5["open"])
        c = safe_float(last5["close"])
        amt5 = safe_float(last5.get("amt"))
        vol5 = safe_float(last5.get("vol"))

        # 5m 成交额基准（动态）
        amt_series_5m = [x for x in df5["amt"].astype(float).tolist() if x is not None]
        amt_ma_5m = ma_dyn(amt_series_5m, 20, min_len=6)

        # 1m 成交额基准（动态；早盘用 ORB 均额兜底）
        last1 = get_last_completed_bar(df1, cutoff) if one_min_available else None
        amt1 = safe_float(last1.get("amt")) if last1 is not None else None
        amt_ma_1m = None
        if one_min_available:
            amt_series_1m = [x for x in df1["amt"].astype(float).tolist() if x is not None]
            amt_ma_1m = ma_dyn(amt_series_1m, 60, min_len=20)
        if amt_ma_1m is None and orb and orb.get("orb_amt_mean_1m") is not None:
            amt_ma_1m = float(orb["orb_amt_mean_1m"])

        events = []

        # ========== ① ORB ==========
        if orb and c is not None:
            buf = float(orb["buf"])
            if c >= float(orb["orb_h"]) * (1 + buf):
                events.append({"type": "ORB_UP_BREAK", "dir": "up",
                               "facts": {"t": fmt_dt(bar_t), "c": c, "orb_h": orb["orb_h"], "buf": buf}})
            if c <= float(orb["orb_l"]) * (1 - buf):
                events.append({"type": "ORB_DN_BREAK", "dir": "down",
                               "facts": {"t": fmt_dt(bar_t), "c": c, "orb_l": orb["orb_l"], "buf": buf}})

        # ========== ② 锚位突破/失守（拆分：只看价格） ==========
        anchor_buf = 0.001  # 10bp
        anchors = []
        if orb:
            anchors += [("ORB_H", orb["orb_h"]), ("ORB_L", orb["orb_l"])]
        if prev:
            anchors += [("PREV_H", prev["high"]), ("PREV_L", prev["low"]), ("PREV_C", prev["close"])]

        if c is not None:
            for name, a in anchors:
                if a is None:
                    continue
                a = float(a)
                if c >= a * (1 + anchor_buf):
                    events.append({"type": f"ANCHOR_BREAK_{name}", "dir": "up",
                                   "facts": {"t": fmt_dt(bar_t), "c": c, "anchor": a, "buf": anchor_buf}})
                if c <= a * (1 - anchor_buf):
                    events.append({"type": f"ANCHOR_LOSE_{name}", "dir": "down",
                                   "facts": {"t": fmt_dt(bar_t), "c": c, "anchor": a, "buf": anchor_buf}})

        # ========== ③ Spike ==========
        if amt5 is not None and amt_ma_5m not in (None, 0):
            ratio5 = amt5 / amt_ma_5m
            if ratio5 >= 3.0:
                events.append({"type": "AMT_SPIKE_5M", "dir": "both",
                               "facts": {"t": fmt_dt(bar_t), "amt5": amt5, "ma_dyn": amt_ma_5m, "ratio": ratio5}})

        if last1 is not None and amt1 is not None and amt_ma_1m not in (None, 0):
            ratio1 = amt1 / amt_ma_1m
            if ratio1 >= 4.0:
                events.append({"type": "AMT_SPIKE_1M", "dir": "both",
                               "facts": {"t": fmt_dt(last1["time"]), "amt1": amt1, "ma_dyn_or_orb": amt_ma_1m, "ratio": ratio1}})

        # ========== 放量确认（拆出来） ==========
        if amt5 is not None and amt_ma_5m not in (None, 0):
            ratio = amt5 / amt_ma_5m
            if ratio >= 2.0:
                events.append({"type": "BREAK_CONFIRM_VOL", "dir": "both",
                               "facts": {"t": fmt_dt(bar_t), "amt5": amt5, "ma_dyn": amt_ma_5m, "ratio": ratio}})

        # ========== ④ 放量滞涨 / 放量下跌 ==========
        ret5 = pct_change(c, o) if (c is not None and o is not None) else None
        if ret5 is not None and amt5 is not None and amt_ma_5m not in (None, 0):
            ratio = amt5 / amt_ma_5m
            if ratio >= 2.5 and abs(ret5) <= 0.2:
                events.append({"type": "VOL_STALL", "dir": "flat",
                               "facts": {"t": fmt_dt(bar_t), "ret5_pct": ret5, "amt_ratio": ratio, "amt5": amt5}})
            if ratio >= 2.0 and ret5 <= -0.6:
                events.append({"type": "VOL_DOWN", "dir": "down",
                               "facts": {"t": fmt_dt(bar_t), "ret5_pct": ret5, "amt_ratio": ratio, "amt5": amt5}})

        # ========== ⑤ 主动占比 proxy ==========
        if one_min_available:
            ap = active_proxy(df1, min_tot_amt_last30=5e6)
            if ap and abs(float(ap["delta"])) >= 0.25:
                events.append({"type": "ACTIVE_PROXY_SHIFT", "dir": "both", "facts": ap})

        # ========== ⑥⑦ 涨停 proxy ==========
        if one_min_available and prev and prev.get("close") is not None:
            lim = estimate_limit_up(sym, float(prev["close"]))
            lp = limitup_proxy(df1, lim)
            if lp:
                events.append({"type": lp["type"], "dir": lp["dir"], "facts": lp["facts"]})

        # ========== ⑧ RS 偏离（按time对齐；指数缺失则本轮跳过RS） ==========
        for idx_code, idxdf in idx_5m.items():
            if idxdf is None or getattr(idxdf, "empty", True):
                continue
            rsz = calc_rs_z_aligned(df5, idxdf, cutoff, window=20)
            if rsz and abs(float(rsz["z"])) >= 2.0:
                events.append({"type": f"RS_DIVERGENCE_{idx_code}",
                               "dir": "up" if rsz["z"] > 0 else "down",
                               "facts": rsz})

        if not events:
            continue

        # ========== 冷却：筛掉本轮不应推送的事件 ==========
        filtered = []
        for e in events:
            k = f"{sym}|{e['type']}|{e['dir']}"
            if should_push(state, k, cooldown_min, now_ts):
                filtered.append((k, e))
        if not filtered:
            continue

        filtered_events = [x[1] for x in filtered]
        filtered_events = sorted(filtered_events, key=event_rank)

        # ========== TopN=5 防刷屏 ==========
        send_events = filtered_events[:TOPN]
        rest_events = filtered_events[TOPN:]

        prompt = build_prompt(
            sym, index_list,
            send_events=send_events,
            rest_events=rest_events,
            context={"now_bj": t.isoformat()}
        )

        # LLM 调用（失败也不致命 + 冷却错误推送）
        try:
            analysis = llm_analyze(os.getenv("OPENAI_BASE_URL","").strip(),
                                   os.getenv("OPENAI_API_KEY","").strip(),
                                   os.getenv("OPENAI_MODEL","deepseek-chat").strip(),
                                   prompt,
                                   max_tokens=int(os.getenv("MAX_TOKENS","450")))
        except Exception as e:
            push_err(sym, "LLM_FAIL", f"【盘中5m监控】{today} {sym}\nLLM调用失败: {repr(e)}")
            continue

        msg = f"【盘中5m监控】{today} {sym} | 事件入LLM={len(send_events)}/{len(filtered_events)}(TopN={TOPN})\n" + analysis
        feishu_send_text(webhook, msg)

        # 仅标记 send_events 进入冷却（避免“未入LLM也冷却导致丢信号”）
        for e in send_events:
            k = f"{sym}|{e['type']}|{e['dir']}"
            mark_push(state, k, now_ts)

    save_state(state)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 最终兜底：不让 workflow 打红（监控不断档）
        print(f"[FATAL_BUT_NONBLOCKING] {repr(e)}")
        raise SystemExit(0)
