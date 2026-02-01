name: intraday-5m-monitor

on:
  schedule:
    - cron: "*/5 1-7 * * 1-5"   # UTC 01-07 每5分钟；脚本内再严格过滤北京时间交易时段
  workflow_dispatch: {}

permissions:
  contents: read

jobs:
  run:
    runs-on: ubuntu-latest
    timeout-minutes: 8

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          pip install -U akshare requests pytz python-dateutil

      - name: Restore intraday state cache
        uses: actions/cache/restore@v4
        with:
          path: intraday_state.json
          key: intraday-state-${{ github.ref_name }}-${{ github.run_id }}
          restore-keys: |
            intraday-state-${{ github.ref_name }}-

      - name: Run intraday watcher
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
          OPENAI_MODEL: ${{ secrets.OPENAI_MODEL }}
          FEISHU_WEBHOOK_URL: ${{ secrets.FEISHU_WEBHOOK_URL }}

          STOCK_LIST: ${{ secrets.STOCK_LIST }}
          INDEX_LIST: ${{ secrets.INDEX_LIST }}
          COOLDOWN_MIN: ${{ secrets.COOLDOWN_MIN }}
          ORB_BUFFER_BP: ${{ secrets.ORB_BUFFER_BP }}
          MAX_TOKENS: ${{ secrets.MAX_TOKENS }}
        run: |
          python intraday_watch_5m.py

      - name: Save intraday state cache
        uses: actions/cache/save@v4
        with:
          path: intraday_state.json
          key: intraday-state-${{ github.ref_name }}-${{ github.run_id }}
