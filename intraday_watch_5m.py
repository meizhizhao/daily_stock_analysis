import os
import requests
from datetime import datetime
from dateutil import tz

BJ = tz.gettz("Asia/Shanghai")

def feishu_send_text(webhook: str, text: str):
    payload = {"msg_type": "text", "content": {"text": text}}
    r = requests.post(webhook, json=payload, timeout=15)
    r.raise_for_status()

def main():
    webhook = os.getenv("FEISHU_WEBHOOK_URL", "").strip()
    stock_list = os.getenv("STOCK_LIST", "").strip()
    if not webhook:
        raise SystemExit("Missing FEISHU_WEBHOOK_URL")
    if not stock_list:
        raise SystemExit("Missing STOCK_LIST")

    t = datetime.now(tz=BJ).strftime("%Y-%m-%d %H:%M:%S %Z")
    feishu_send_text(webhook, f"【intraday-5m-monitor 测试】{t}\nSTOCK_LIST={stock_list}\n脚本已成功运行。")

if __name__ == "__main__":
    main()
