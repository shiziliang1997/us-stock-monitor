# -*- coding: utf-8 -*-

import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

try:
    from IPython.display import display
except ImportError:
    display = print


# =========================
# PushPlus 微信推送配置
# =========================
# Token 从 GitHub Secret 读取，不要写死在代码里
PUSHPLUS_TOKEN = os.getenv("PUSHPLUS_TOKEN")


def send_wechat_push(title, content):
    """
    使用 PushPlus 微信渠道推送消息
    """
    if not PUSHPLUS_TOKEN:
        print("未配置 PUSHPLUS_TOKEN，跳过微信推送")
        return

    url = "https://www.pushplus.plus/send"

    data = {
        "token": PUSHPLUS_TOKEN,
        "title": title,
        "content": content,
        "template": "markdown"
    }

    try:
        response = requests.post(url, json=data, timeout=10)
        print("微信推送状态码：", response.status_code)
        print("微信推送结果：", response.text)
    except Exception as e:
        print("微信推送失败：", e)


# =========================
# 最新投资组合
# =========================

WATCHLIST = {
    "NVDA": {
        "name": "英伟达",
        "type": "core_ai",
        "target_weight": 0.30,
        "max_weight": 0.30
    },
    "GOOGL": {
        "name": "Alphabet",
        "type": "core_ai",
        "target_weight": 0.20,
        "max_weight": 0.20
    },
    "MU": {
        "name": "美光",
        "type": "aggressive_memory",
        "target_weight": 0.15,
        "max_weight": 0.15
    },
    "TSM": {
        "name": "台积电",
        "type": "core_semiconductor",
        "target_weight": 0.14,
        "max_weight": 0.14
    },
    "META": {
        "name": "Meta",
        "type": "core_ai",
        "target_weight": 0.11,
        "max_weight": 0.11
    },
    "AMZN": {
        "name": "亚马逊",
        "type": "core_cloud_ai",
        "target_weight": 0.10,
        "max_weight": 0.10
    },
}

MARKET_TICKERS = ["QQQ", "SPY", "SOXX", "^VIX", "^TNX"]
ALL_TICKERS = list(WATCHLIST.keys()) + MARKET_TICKERS

TOTAL_CAPITAL_RMB = 50000


# =========================
# 数据处理函数
# =========================

def clean_yfinance_df(df, ticker):
    """
    修复 yfinance 在 GitHub Actions / Colab / Jupyter 中可能出现的多层列问题
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    df.columns = [str(c).strip() for c in df.columns]

    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    needed_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing_cols = [c for c in needed_cols if c in df.columns]
    df = df[existing_cols]

    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])

    return df


def download_data(tickers, period="2y"):
    data = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column"
            )

            df = clean_yfinance_df(df, ticker)

            if df.empty or "Close" not in df.columns:
                print(f"警告：{ticker} 数据无效，无法识别 Close 列")
                continue

            data[ticker] = df

        except Exception as e:
            print(f"下载 {ticker} 失败：{e}")

    return data


def add_indicators(df):
    df = df.copy()

    df["MA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=50).mean()
    df["MA200"] = df["Close"].rolling(200, min_periods=200).mean()
    df["VOL20"] = df["Volume"].rolling(20, min_periods=20).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=14).mean()

    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["Drawdown_20d"] = df["Close"] / df["Close"].rolling(20, min_periods=20).max() - 1

    return df


# =========================
# 市场环境判断
# =========================

def market_signal(data):
    signals = {}

    for ticker in ["QQQ", "SPY", "SOXX"]:
        if ticker not in data:
            signals[ticker] = "数据缺失，无法判断"
            continue

        df = add_indicators(data[ticker]).dropna(subset=["Close", "MA50"])

        if df.empty:
            signals[ticker] = "数据不足，无法判断"
            continue

        last = df.iloc[-1]
        close = float(last["Close"])
        ma50 = float(last["MA50"])

        if close > ma50:
            signals[ticker] = "健康：在50日线上方"
        else:
            signals[ticker] = "风险：跌破50日线"

    if "^VIX" in data and not data["^VIX"].dropna(subset=["Close"]).empty:
        vix = float(data["^VIX"].dropna(subset=["Close"]).iloc[-1]["Close"])
    else:
        vix = np.nan

    if "^TNX" in data and not data["^TNX"].dropna(subset=["Close"]).empty:
        tnx = float(data["^TNX"].dropna(subset=["Close"]).iloc[-1]["Close"])
    else:
        tnx = np.nan

    if pd.isna(vix):
        vix_signal = "VIX数据缺失"
    elif vix < 20:
        vix_signal = "低波动"
    elif vix < 25:
        vix_signal = "波动升温"
    else:
        vix_signal = "恐慌"

    market_risk_score = 0

    if "风险" in signals.get("QQQ", ""):
        market_risk_score += 1

    if "风险" in signals.get("SOXX", ""):
        market_risk_score += 1

    if not pd.isna(vix) and vix >= 25:
        market_risk_score += 1

    if market_risk_score == 0:
        market_status = "绿灯：可以进攻，允许分批加仓"
    elif market_risk_score == 1:
        market_status = "黄灯：谨慎，暂停追涨"
    else:
        market_status = "红灯：防守，降低MU等高波动仓位"

    return {
        "QQQ": signals.get("QQQ", "无法判断"),
        "SPY": signals.get("SPY", "无法判断"),
        "SOXX": signals.get("SOXX", "无法判断"),
        "VIX": round(vix, 2) if not pd.isna(vix) else "数据缺失",
        "VIX_signal": vix_signal,
        "TNX_10Y": round(tnx, 2) if not pd.isna(tnx) else "数据缺失",
        "market_status": market_status
    }


# =========================
# 个股信号判断
# =========================

def stock_signal(ticker, df, stock_type):
    df = add_indicators(df)

    df = df.dropna(
        subset=[
            "Close",
            "MA20",
            "MA50",
            "MA200",
            "RSI14",
            "Drawdown_20d",
            "VOL20"
        ]
    )

    name = WATCHLIST.get(ticker, {}).get("name", "")

    if df.empty:
        return {
            "股票": ticker,
            "名称": name,
            "价格": "数据不足",
            "20日线": "数据不足",
            "50日线": "数据不足",
            "200日线": "数据不足",
            "RSI14": "数据不足",
            "20日回撤": "数据不足",
            "趋势": "无法判断",
            "信号灯": "灰灯",
            "风险等级": "数据异常",
            "风险提示": "历史数据不足或下载异常",
            "操作建议": "暂不根据程序调仓"
        }

    last = df.iloc[-1]

    close = float(last["Close"])
    ma20 = float(last["MA20"])
    ma50 = float(last["MA50"])
    ma200 = float(last["MA200"])
    rsi = float(last["RSI14"])
    volume = float(last["Volume"])
    vol20 = float(last["VOL20"])
    drawdown_20d = float(last["Drawdown_20d"])

    warnings = []
    actions = []

    is_aggressive = "aggressive" in stock_type

    if close > ma20 > ma50:
        trend = "强势上升"
        light = "绿灯"
        actions.append("继续持有；未满目标仓位可等待回踩分批加仓")

    elif close > ma50:
        trend = "趋势尚可"
        light = "绿灯偏黄"
        actions.append("持有，不追高")

    elif close < ma50 and close > ma200:
        trend = "跌破50日线"
        light = "黄灯"
        warnings.append("中期趋势转弱")
        actions.append("暂停加仓；高波动仓位考虑减仓1/3")

    else:
        trend = "跌破200日线"
        light = "红灯"
        warnings.append("大级别趋势转弱")
        actions.append("降低仓位，等待重新站上50日线")

    if close < ma50 and volume > 1.5 * vol20:
        warnings.append("放量跌破50日线")
        actions.append("优先减仓，避免继续补仓摊薄")

    if rsi > 75:
        warnings.append("RSI过热，短期追高风险")
        actions.append("不要一次性加仓，等待回调")

    elif rsi < 35:
        warnings.append("RSI偏低，可能超跌")
        actions.append("只在未破位且基本面没坏时考虑小幅加仓")

    if drawdown_20d < -0.12:
        warnings.append("20日内回撤超过12%")

        if is_aggressive:
            actions.append("高波动进攻仓需要检查是否触发止损")
        else:
            actions.append("核心仓可观察是否跌破50日线")

    if is_aggressive:
        if close < ma50:
            actions.append("进攻仓规则：跌破50日线，建议减仓1/3")
        if drawdown_20d < -0.15:
            actions.append("进攻仓规则：短期回撤超过15%，不建议继续补仓")

    if light == "红灯":
        risk_level = "S级风险"
    elif "放量跌破50日线" in warnings or trend == "跌破50日线":
        risk_level = "A级风险"
    elif drawdown_20d < -0.12:
        risk_level = "B级风险"
    elif rsi > 75 or rsi < 35:
        risk_level = "C级风险"
    else:
        risk_level = "正常"

    return {
        "股票": ticker,
        "名称": name,
        "价格": round(close, 2),
        "20日线": round(ma20, 2),
        "50日线": round(ma50, 2),
        "200日线": round(ma200, 2),
        "RSI14": round(rsi, 1),
        "20日回撤": f"{drawdown_20d:.2%}",
        "趋势": trend,
        "信号灯": light,
        "风险等级": risk_level,
        "风险提示": "；".join(warnings) if warnings else "无明显风险",
        "操作建议": "；".join(dict.fromkeys(actions))
    }


# =========================
# 目标组合
# =========================

def portfolio_plan():
    rows = []

    for ticker, info in WATCHLIST.items():
        target_amount = TOTAL_CAPITAL_RMB * info["target_weight"]
        max_amount = TOTAL_CAPITAL_RMB * info["max_weight"]

        rows.append({
            "股票": ticker,
            "名称": info["name"],
            "类型": info["type"],
            "目标仓位": f"{info['target_weight']:.0%}",
            "目标金额_人民币": round(target_amount, 0),
            "最大仓位": f"{info['max_weight']:.0%}",
            "最大金额_人民币": round(max_amount, 0)
        })

    return pd.DataFrame(rows)


# =========================
# 日报优化函数
# =========================

def portfolio_status(result_df, mkt):
    red_count = 0
    yellow_count = 0
    green_count = 0

    for _, row in result_df.iterrows():
        light = str(row["信号灯"])

        if "红灯" in light:
            red_count += 1
        elif "黄灯" in light:
            yellow_count += 1
        elif "绿灯" in light:
            green_count += 1

    market_status = str(mkt.get("market_status", ""))

    if red_count >= 2 or "红灯" in market_status:
        status = "红灯"
        conclusion = "组合风险较高，应降低进攻仓位，暂停加仓，优先保护本金。"
    elif red_count >= 1 or yellow_count >= 2 or "黄灯" in market_status:
        status = "黄灯"
        conclusion = "组合出现局部风险，建议暂停追涨，重点观察破位个股。"
    else:
        status = "绿灯"
        conclusion = "组合整体健康，可继续持有；未满目标仓位可等待回踩分批加仓。"

    return status, conclusion


def action_summary(result_df):
    actions = []

    for _, row in result_df.iterrows():
        ticker = row["股票"]
        name = row["名称"]
        light = str(row["信号灯"])
        trend = str(row["趋势"])
        risk = str(row["风险提示"])

        if "红灯" in light:
            actions.append(f"{ticker} {name}：红灯，优先控制风险；等待重新站上50日线后再考虑加仓。")

        elif "跌破50日线" in trend:
            actions.append(f"{ticker} {name}：跌破50日线，暂停加仓；若连续无法收复，考虑减仓1/3。")

        elif "RSI过热" in risk:
            actions.append(f"{ticker} {name}：RSI过热，不建议追高；等待回调或横盘消化。")

        elif "20日内回撤超过12%" in risk:
            actions.append(f"{ticker} {name}：短期回撤较大，先观察是否守住50日线。")

        elif "绿灯" in light:
            actions.append(f"{ticker} {name}：趋势健康，继续持有；不要情绪化卖出。")

        else:
            actions.append(f"{ticker} {name}：继续观察，暂不主动加仓。")

    return actions


def major_risk_rows(result_df):
    alert_rows = []

    for _, row in result_df.iterrows():
        text = (
            str(row["信号灯"])
            + str(row["趋势"])
            + str(row["风险等级"])
            + str(row["风险提示"])
            + str(row["操作建议"])
        )

        if (
            "红灯" in text
            or "跌破50日线" in text
            or "放量跌破50日线" in text
            or "减仓" in text
            or "止损" in text
            or "不建议继续补仓" in text
            or "S级风险" in text
            or "A级风险" in text
        ):
            alert_rows.append(row)

    return alert_rows


def build_daily_report(result_df, mkt, plan_df):
    status, conclusion = portfolio_status(result_df, mkt)
    actions = action_summary(result_df)

    report_content = f"""## 美股监控日报｜{datetime.now().strftime('%Y-%m-%d %H:%M')}

### 一、今日总判断

- 组合状态：{status}
- 核心结论：{conclusion}

---

### 二、市场环境

"""

    if mkt:
        for k, v in mkt.items():
            report_content += f"- {k}：{v}\n"
    else:
        report_content += "- 市场环境数据生成失败\n"

    report_content += "\n---\n\n"
    report_content += "### 三、个股监控\n\n"

    for _, row in result_df.iterrows():
        report_content += f"""
#### {row['股票']} {row['名称']}｜{row['信号灯']}｜{row['风险等级']}

- 价格：{row['价格']}
- 趋势：{row['趋势']}
- 20日线：{row['20日线']}
- 50日线：{row['50日线']}
- 200日线：{row['200日线']}
- RSI14：{row['RSI14']}
- 20日回撤：{row['20日回撤']}
- 风险提示：{row['风险提示']}
- 操作建议：{row['操作建议']}

---
"""

    report_content += "\n### 四、今日操作清单\n\n"

    for i, action in enumerate(actions, 1):
        report_content += f"{i}. {action}\n"

    report_content += "\n---\n\n"
    report_content += "### 五、目标组合\n\n"

    for _, row in plan_df.iterrows():
        report_content += (
            f"- {row['股票']} {row['名称']}："
            f"目标仓位 {row['目标仓位']}，"
            f"目标金额约 {int(row['目标金额_人民币'])} 元\n"
        )

    report_content += "\n---\n\n"
    report_content += "### 六、纪律提醒\n\n"
    report_content += "- 不因为单日波动冲动交易。\n"
    report_content += "- 绿灯不代表追高，红灯不代表恐慌清仓。\n"
    report_content += "- 跌破50日线优先暂停加仓；放量破位再考虑减仓。\n"
    report_content += "- 所有信号仅用于辅助决策，最终仍需结合基本面和个人风险承受能力。\n"

    return report_content


def build_risk_report(result_df, mkt):
    alert_rows = major_risk_rows(result_df)
    status, conclusion = portfolio_status(result_df, mkt)

    report_content = f"""## 美股晚间风险扫描｜{datetime.now().strftime('%Y-%m-%d %H:%M')}

### 一、组合状态

- 组合状态：{status}
- 核心结论：{conclusion}

---

### 二、市场环境

"""

    if mkt:
        for k, v in mkt.items():
            report_content += f"- {k}：{v}\n"
    else:
        report_content += "- 市场环境数据生成失败\n"

    report_content += "\n---\n\n"

    if len(alert_rows) == 0:
        report_content += "### 三、风险扫描结果\n\n"
        report_content += "- 当前未发现重大风险信号。\n"
        report_content += "- 继续持有，避免无意义频繁操作。\n"
        return report_content

    report_content += "### 三、触发风险的标的\n\n"

    for row in alert_rows:
        report_content += f"""
#### {row['股票']} {row['名称']}｜{row['信号灯']}｜{row['风险等级']}

- 价格：{row['价格']}
- 趋势：{row['趋势']}
- 50日线：{row['50日线']}
- RSI14：{row['RSI14']}
- 20日回撤：{row['20日回撤']}
- 风险提示：{row['风险提示']}
- 操作建议：{row['操作建议']}

---
"""

    return report_content


# =========================
# 主程序
# =========================

def main():
    print("=" * 90)
    print(f"美股监控报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    data = download_data(ALL_TICKERS)

    print("\n已成功下载以下数据：")
    print(list(data.keys()))

    print("\n一、市场环境")
    print("-" * 90)

    mkt = {}

    try:
        mkt = market_signal(data)
        for k, v in mkt.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"市场环境判断失败：{e}")

    print("\n二、个股监控")
    print("-" * 90)

    result = []

    for ticker, info in WATCHLIST.items():
        if ticker not in data:
            result.append({
                "股票": ticker,
                "名称": info["name"],
                "价格": "下载失败",
                "20日线": "下载失败",
                "50日线": "下载失败",
                "200日线": "下载失败",
                "RSI14": "下载失败",
                "20日回撤": "下载失败",
                "趋势": "无法判断",
                "信号灯": "灰灯",
                "风险等级": "数据异常",
                "风险提示": "yfinance未成功获取数据",
                "操作建议": "暂不根据程序调仓"
            })
            continue

        signal = stock_signal(ticker, data[ticker], info["type"])
        result.append(signal)

    result_df = pd.DataFrame(result)
    display(result_df)

    print("\n三、目标组合")
    print("-" * 90)

    plan_df = portfolio_plan()
    display(plan_df)

    result_df.to_csv("us_stock_monitor_result.csv", index=False, encoding="utf-8-sig")
    plan_df.to_csv("us_stock_portfolio_plan.csv", index=False, encoding="utf-8-sig")

    print("\n已生成文件：")
    print("1. us_stock_monitor_result.csv")
    print("2. us_stock_portfolio_plan.csv")

    # =========================
    # 微信推送逻辑
    # =========================
    # 如果你的 yml 里设置了：
    # GITHUB_SCHEDULE: ${{ github.event.schedule }}
    # 则：
    # 0 1 * * 1-5  = 北京时间09:00，推送完整日报
    # 0 13 * * 1-5 = 北京时间21:00，推送风险扫描
    #
    # 如果没有设置 GITHUB_SCHEDULE，默认推送完整日报。

github_schedule = os.getenv("GITHUB_SCHEDULE", "")

if github_schedule in [
    "25 14 * * 1-5",
    "30 14 * * 1-5",
    "35 14 * * 1-5",
    "23 15 * * 1-5",
]:
    report_content = build_risk_report(result_df, mkt)
    send_wechat_push("美股晚间风险扫描", report_content)
else:
    report_content = build_daily_report(result_df, mkt, plan_df)
    send_wechat_push("美股监控日报", report_content)


if __name__ == "__main__":
    main()
