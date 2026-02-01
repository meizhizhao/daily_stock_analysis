import os
import json
import math
import time
from datetime import datetime, timedelta
from dateutil import tz

import requests

# ---- Optional: AKShare (installed in workflow) ----
import akshare as ak

BJ = tz.gettz("Asia/Shanghai")


# =========================
# Basic helpers
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
STATE_PATH = "intraday_state.json"

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
# DeepSeek (OpenAI-compatible)
# =========================
def llm_analyze(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int):
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content":
                "你是一名A股盘中监控与交易分析助手。"
                "必须输出可复核结论：每条结论必须包含“日期时间+数值”证据。"
                "⑤⑥⑦属于proxy近似，必须明确标注不可替代L2逐笔/盘口。"
                "输出结构固定："
                "1) 一句话结论(<=30字)"
                "2) 证据表(>=5条，日期时间+数值)"
                "3) 推理链(基于证据，不许空话)"
                "4) 行动建议(触发条件/无效条件/风控线，必须给数值)"
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=body, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# =========================
# AKShare fetch + normalize
# =========================
def _parse_time_any(s: str) -> datetime:
    # AKShare常见格式: "YYYY-MM-DD HH:MM:SS"
    return datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S").replace(tzinfo=BJ)

def normalize_min_df(df):
    """
    标准化成列：time, open, high, low, close, vol, amt
    """
    if df is None or df.empty:
        return None
    cols = set(df.columns)

    # 常见中文列名
    time_col = "时间" if "时间" in cols else ("日期" if "日期" in cols else None)
    open_col = "开盘" if "开盘" in cols else None
    high_col = "最高" if "最高" in cols else None
    low_col  = "最低" if "最低" in cols else None
    close_col= "收盘" if "收盘" in cols else ("最新价" if "最新价" in cols else None)
    vol_col  = "成交量" if "成交量" in cols else None
    amt_col  = "成交额" if "成交额" in cols else None

    if not time_col or not open_col or not high_col or not low_col or not close_col:
        # 列不全就直接失败，让日志告诉我们
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    out = df[[time_col, open_col, high_col, low_col, close_col]].copy()
    out.columns = ["time", "open", "high", "low", "close"]
    if vol_col:
        out["vol"] = df[vol_col].apply(safe_float)
    else:
        out["vol"] = None
    if amt_col:
        out["amt"] = df[amt_col].apply(safe_float)
    else:
        out["amt"] = None

    # time 转 datetime
    if not isinstance(out["time"].iloc[0], datetime):
        out["time"] = out["time"].apply(_parse_time_any)

    return out.sort_values("time")

def fetch_stock_min(symbol: str, period: str, start_dt: str, end_dt: str):
    def _fn():
        return ak.stock_zh_a_hist_min_em(symbol=symbol, period=period, start_date=start_dt, end_date=end_dt, adjust="")
    return retry(_fn, desc=f"stock_min {symbol} {period}")

def fetch_index_min(symbol: str, period: str, start_dt: str, end_dt: str):
    def _fn():
        return ak.index_zh_a_hist_min_em(symbol=symbol, period=period, start_date=start_dt, end_date=end_dt)
    return retry(_fn, desc=f"index_min {symbol} {period}")

def fetch_prev_day_ohlc(symbol: str, yyyymmdd: str):
    # 拉近60天日线，取倒数第二根作为“昨日”
    end_date = datetime.strptime(yyyymmdd, "%Y%m%d").date()
    start_date = (end_date - timedelta(days=60)).strftime("%Y%m%d")
    def _fn():
        return ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=yyyymmdd, adjust="")
    df = retry(_fn, desc=f"daily {symbol}")
    if df is None or df.empty or "日期" not in df.columns or len(df) < 2:
        return None
    df = df.sort_values("日期")
    prev = df.iloc[-2]
    return {
        "date": str(prev["日期"]),
        "open": safe_float(prev.get("开盘")),
        "high": safe_float(prev.get("最高")),
        "low": safe_float(prev.get("最低")),
        "close": safe_float(prev.get("收盘")),
    }


# =========================
# Signals
# =========================
def get_last_completed_bar(df_norm, cutoff: datetime):
    # 取 <= cutoff 的最后一根
    d2 = df_norm[df_norm["time"] <= cutoff]
    if d2.empty:
        return None
    return d2.iloc[-1]

def calc_orb(one_min_norm, orb_buffer_bp: int):
    if one_min_norm is None or one_min_norm.empty:
        return None
    t0 = one_min_norm["time"].iloc[0]
    day = t0.date()
    s = datetime(day.year, day.month, day.day, 9, 30, tzinfo=BJ)
    e = datetime(day.year, day.month, day.day, 9, 35, tzinfo=BJ)  # [09:30,09:35)
    w = one_min_norm[(one_min_norm["time"] >= s) & (one_min_norm["time"] < e)]
    if w.empty or len(w) < 3:
        return None
    orb_h = float(w["high"].max())
    orb_l = float(w["low"].min())
    buf = orb_buffer_bp / 10000.0
    return {"orb_h": orb_h, "orb_l": orb_l, "buf": buf}

def estimate_limit_up(symbol: str, prev_close: float):
    # proxy：300/688 20%，其余 10%（未处理ST/北交所等）
    pct = 0.2 if (symbol.startswith("300") or symbol.startswith("688")) else 0.1
    return round(prev_close * (1 + pct), 2)

def limitup_proxy(one_min_norm, limit_up: float):
    # proxy：分钟级触板/封板/炸板/回封耗时
    tol = 0.01
    df = one_min_norm
    touched = df[df["high"] >= limit_up - tol]
    if touched.empty:
        return None
    first_touch = touched["time"].iloc[0]

    sealed = df[df["close"] >= limit_up - tol]
    if sealed.empty:
        return {"type": "LIMIT_TOUCH", "dir": "up",
                "facts": {"t_first_touch": fmt_dt(first_touch), "limit_up": limit_up, "note": "proxy:分钟级触板"}}

    first_seal = sealed["time"].iloc[0]
    after = df[df["time"] >= first_seal]
    opened = after[after["close"] < limit_up - tol]
    if opened.empty:
        # 持续封板分钟数
        seal_mins = int((after["close"] >= limit_up - tol).sum())
        return {"type": "LIMIT_SEALED", "dir": "up",
                "facts": {"t_first_seal": fmt_dt(first_seal), "limit_up": limit_up, "seal_minutes_proxy": seal_mins,
                          "note": "proxy:封板持续分钟数(非封单金额/撤单率)"}}

    first_open = opened["time"].iloc[0]
    after_open = after[after["time"] >= first_open]
    reseal = after_open[after_open["close"] >= limit_up - tol]
    if reseal.empty:
        return {"type": "LIMIT_BREAK", "dir": "down",
                "facts": {"t_first_seal": fmt_dt(first_seal), "t_first_open": fmt_dt(first_open), "limit_up": limit_up,
                          "note": "proxy:炸板后未回封(分钟级)"}}

    reseal_t = reseal["time"].iloc[0]
    minutes = int((reseal_t - first_open).total_seconds() / 60)
    return {"type": "LIMIT_RESEAL", "dir": "up",
            "facts": {"t_first_open": fmt_dt(first_open), "t_reseal": fmt_dt(reseal_t), "reseal_minutes_proxy": minutes,
                      "limit_up": limit_up, "note": "proxy:回封耗时按分钟(非逐笔秒级)"}}

def active_proxy(one_min_norm):
    # proxy：上涨K线成交额占比（近30m vs 前60m）
    df = one_min_norm
    cutoff = now_bj() - timedelta(minutes=1)
    df = df[df["time"] <= cutoff]
    if len(df) < 120:
        return None

    def up_amt_ratio(w):
        up = w[w["close"] > w["open"]]["amt"].sum()
        tot = w["amt"].sum()
        return float(up / tot) if tot else None

    last30 = df.iloc[-30:]
    prev60 = df.iloc[-90:-30]
    r1 = up_amt_ratio(last30)
    r0 = up_amt_ratio(prev60)
    if r1 is None or r0 is None:
        return None

    delta = r1 - r0
    return {"t": fmt_dt(df["time"].iloc[-1]),
            "up_amt_ratio_last30": r1, "up_amt_ratio_prev60": r0, "delta": delta,
            "note": "proxy:上涨K线成交额占比近似主动买盘占比"}

def build_prompt(symbol: str, idx_list: str, events: list, context: dict):
    lines = []
    lines.append(f"标的: {symbol}")
    lines.append(f"北京时间: {context['now_bj']}")
    lines.append(f"基准指数: {idx_list}")
    lines.append("")
    lines.append("触发事件（证据必须带日期时间+数值）：")
    for e in events:
        lines.append(f"- {e['type']} | dir={e['dir']} | facts={json.dumps(e['facts'], ensure_ascii=False)}")
    lines.append("")
    lines.append("注意：⑤⑥⑦为proxy近似，必须显式标注，不可当L2逐笔/盘口结论。")
    return "\n".join(lines)


def main():
    # ---------- env ----------
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
        # 非交易时段：直接退出，不推送（避免刷屏）
        print(f"Not in trading session: {t.isoformat()}; exit.")
        return

    today = t.strftime("%Y-%m-%d")
    start_dt = f"{today} 09:30:00"
    end_dt = f"{today} 15:00:00"
    yyyymmdd = t.strftime("%Y%m%d")
    cutoff = t - timedelta(minutes=1)

    # ---------- state ----------
    state = load_state()
    if state.get("date") != today:
        state = {"date": today, "orb": {}, "last_push": {}}

    # ---------- index 5m ----------
    idx_5m = {}
    for idx in idx_codes:
        try:
            raw = fetch_index_min(idx, "5", start_dt, end_dt)
            idx_5m[idx] = normalize_min_df(raw)
        except Exception as e:
            print(f"Index fetch failed {idx}: {repr(e)}")

    # ---------- per stock ----------
    for sym in symbols:
        events = []
        try:
            raw_1m = fetch_stock_min(sym, "1", start_dt, end_dt)
            raw_5m = fetch_stock_min(sym, "5", start_dt, end_dt)
            df1 = normalize_min_df(raw_1m)
            df5 = normalize_min_df(raw_5m)
        except Exception as e:
            feishu_send_text(webhook, f"【盘中5m监控】{today} {sym}\n数据拉取失败: {repr(e)}")
            continue

        last5 = get_last_completed_bar(df5, cutoff)
        if last5 is None:
            continue

        # series
        amt_series_5m = [x for x in df5["amt"].tolist() if x is not None]
        vol_series_5m = [x for x in df5["vol"].tolist() if x is not None]
        amt_ma20_5m = ma(amt_series_5m, 20)
        vol_ma20_5m = ma(vol_series_5m, 20)

        # ① ORB
        orb = calc_orb(df1, orb_buffer_bp)
        if orb:
            state.setdefault("orb", {})[sym] = orb
        orb = state.get("orb", {}).get(sym)

        c = safe_float(last5["close"])
        o = safe_float(last5["open"])
        h = safe_float(last5["high"])
        l = safe_float(last5["low"])
        amt = safe_float(last5["amt"])
        vol = safe_float(last5["vol"])
        bar_t = last5["time"]

        if orb and c is not None:
            buf = orb["buf"]
            if c >= orb["orb_h"] * (1 + buf):
                events.append({"type": "ORB_UP_BREAK", "dir": "up",
                               "facts": {"t": fmt_dt(bar_t), "c": c, "orb_h": orb["orb_h"], "buf": buf}})
            if c <= orb["orb_l"] * (1 - buf):
                events.append({"type": "ORB_DN_BREAK", "dir": "down",
                               "facts": {"t": fmt_dt(bar_t), "c": c, "orb_l": orb["orb_l"], "buf": buf}})

        # ② Anchors（ORB + prev day）
        prev = None
        try:
            prev = fetch_prev_day_ohlc(sym, yyyymmdd)
        except Exception as e:
            print(f"Prev day fetch failed {sym}: {repr(e)}")

        anchors = []
        if orb:
            anchors += [("ORB_H", orb["orb_h"]), ("ORB_L", orb["orb_l"])]
        if prev:
            anchors += [("PREV_H", prev["high"]), ("PREV_L", prev["low"]), ("PREV_C", prev["close"])]

        anchor_buf = 0.001
        for name, a in anchors:
            if a is None or c is None:
                continue
            if c >= a * (1 + anchor_buf) and (amt_ma20_5m is None or (amt is not None and amt >= amt_ma20_5m)):
                events.append({"type": f"ANCHOR_BREAK_{name}", "dir": "up",
                               "facts": {"t": fmt_dt(bar_t), "c": c, "anchor": a, "buf": anchor_buf, "amt": amt, "amt_ma20_5m": amt_ma20_5m}})
            if c <= a * (1 - anchor_buf) and (amt_ma20_5m is None or (amt is not None and amt >= amt_ma20_5m)):
                events.append({"type": f"ANCHOR_LOSE_{name}", "dir": "down",
                               "facts": {"t": fmt_dt(bar_t), "c": c, "anchor": a, "buf": anchor_buf, "amt": amt, "amt_ma20_5m": amt_ma20_5m}})

        # ③ Volume Spike（1m/5m）
        # 5m spike
        if amt is not None and amt_ma20_5m not in (None, 0):
            ratio5 = amt / amt_ma20_5m
            if ratio5 >= 3.0:
                events.append({"type": "AMT_SPIKE_5M", "dir": "both",
                               "facts": {"t": fmt_dt(bar_t), "amt_5m": amt, "amt_ma20_5m": amt_ma20_5m, "ratio": ratio5}})
        # 1m spike（用最后1m已收盘）
        last1 = get_last_completed_bar(df1, cutoff)
        if last1 is not None:
            amt1 = safe_float(last1["amt"])
            amt_series_1m = [x for x in df1["amt"].tolist() if x is not None]
            amt_ma60_1m = ma(amt_series_1m, 60)
            if amt1 is not None and amt_ma60_1m not in (None, 0):
                ratio1 = amt1 / amt_ma60_1m
                if ratio1 >= 4.0:
                    events.append({"type": "AMT_SPIKE_1M", "dir": "both",
                                   "facts": {"t": fmt_dt(last1["time"]), "amt_1m": amt1, "amt_ma60_1m": amt_ma60_1m, "ratio": ratio1}})

        # ④ 放量滞涨 / 放量下跌（5m）
        ret5 = pct_change(c, o) if (c is not None and o is not None) else None
        if ret5 is not None and amt is not None and amt_ma20_5m not in (None, 0):
            ratio = amt / amt_ma20_5m
            if ratio >= 2.5 and abs(ret5) <= 0.2:
                events.append({"type": "VOL_STALL", "dir": "flat",
                               "facts": {"t": fmt_dt(bar_t), "ret_5m_pct": ret5, "amt_ratio": ratio}})
            if ratio >= 2.0 and ret5 <= -0.6:
                events.append({"type": "VOL_DOWN", "dir": "down",
                               "facts": {"t": fmt_dt(bar_t), "ret_5m_pct": ret5, "amt_ratio": ratio}})

        # ⑤ 主动买卖占比突变（proxy）
        ap = active_proxy(df1)
        if ap and abs(ap["delta"]) >= 0.25:
            events.append({"type": "ACTIVE_PROXY_SHIFT", "dir": "both", "facts": ap})

        # ⑥⑦ 涨停封板/炸板/回封耗时 & 封单撤单 proxy
        if prev and prev.get("close") is not None:
            lim = estimate_limit_up(sym, prev["close"])
            lup = limitup_proxy(df1, lim)
            if lup:
                events.append({"type": lup["type"], "dir": lup["dir"], "facts": lup["facts"]})

        # ⑧ 相对指数强弱偏离（5m）
        if o is not None and c is not None and o != 0:
            stock_ret = c / o - 1.0
            for idx_code, idxdf in idx_5m.items():
                if idxdf is None or idxdf.empty:
                    continue
                ilast = get_last_completed_bar(idxdf, cutoff)
                if ilast is None:
                    continue
                io = safe_float(ilast["open"])
                ic = safe_float(ilast["close"])
                if io in (None, 0) or ic is None:
                    continue
                idx_ret = ic / io - 1.0
                rs = stock_ret - idx_ret

                # zscore on rolling 20 of rs series（简化：用同长度末尾对齐）
                # 构造 stock_ret_series / idx_ret_series
                s_open = [x for x in df5["open"].tolist() if x is not None]
                s_close= [x for x in df5["close"].tolist() if x is not None]
                i_open = [x for x in idxdf["open"].tolist() if x is not None]
                i_close= [x for x in idxdf["close"].tolist() if x is not None]
                n = min(len(s_open), len(s_close), len(i_open), len(i_close))
                if n >= 25:
                    s_ret_series = [(s_close[-k] / s_open[-k] - 1.0) for k in range(n, 0, -1)]
                    i_ret_series = [(i_close[-k] / i_open[-k] - 1.0) for k in range(n, 0, -1)]
                    rs_series = [s_ret_series[j] - i_ret_series[j] for j in range(n)]
                    mu = ma(rs_series, 20)
                    sd = std(rs_series, 20)
                    if mu is not None and sd not in (None, 0):
                        z = (rs - mu) / sd
                        if abs(z) >= 2.0:
                            events.append({"type": f"RS_DIVERGENCE_{idx_code}",
                                           "dir": "up" if z > 0 else "down",
                                           "facts": {"t": fmt_dt(bar_t),
                                                     "stock_ret_5m": stock_ret,
                                                     "idx_ret_5m": idx_ret,
                                                     "rs": rs,
                                                     "rs_z20": z}})

        # ---- event-driven push + cooldown ----
        if not events:
            continue

        # 去重冷却：按“股票|事件类型|方向”
        now_ts = time.time()
        fire = []
        for e in events:
            k = f"{sym}|{e['type']}|{e['dir']}"
            if should_push(state, k, cooldown_min, now_ts):
                fire.append((k, e))

        if not fire:
            continue

        fired_events = [x[1] for x in fire]
        prompt = build_prompt(sym, index_list, fired_events, {"now_bj": t.isoformat()})

        try:
            analysis = llm_analyze(base_url, api_key, model, prompt, max_tokens=max_tokens)
        except Exception as e:
            feishu_send_text(webhook, f"【盘中5m监控】{today} {sym}\nLLM调用失败: {repr(e)}")
            continue

        msg = f"【盘中5m监控】{today} {sym} | 触发{len(fired_events)}项\n" + analysis
        feishu_send_text(webhook, msg)

        for k, _ in fire:
            mark_push(state, k, now_ts)

    save_state(state)


if __name__ == "__main__":
    main()
