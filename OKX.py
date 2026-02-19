#!/usr/bin/env python3
"""
OKX K线数据重采样、下载、合并、指标计算与AI分析工具
功能：
  1. 从历史文件夹读取1分钟K线，重采样为目标周期
  2. 下载最新目标周期数据并合并
  3. 计算技术指标
  4. 调用DeepSeek API生成分析报告
  5. 交互式对话，支持仓位计算器
  6. 保存报告和对话记录到D:/ceshi
"""

import os
import sys
import math
import time
import requests
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from openai import OpenAI

# ==================== 配置区域 ====================
class Config:
    # ----- 路径设置 -----
    HISTORY_DIR = "D:/ceshi2"          # 存放历史1分钟K线CSV的文件夹
    OUTPUT_DIR = "D:/ceshi"                   # 输出目录（所有文件保存在这里）

    # ----- 交易对与周期 -----
    SYMBOL = None                              # 在运行时由用户输入
    SOURCE_TIMEFRAME = "1m"                    # 历史数据的原始周期（必须与文件一致）
    TARGET_TIMEFRAME = "15m"                    # 目标周期：例如 "15m", "1H", "4H", "1D" 等
    DAYS = 3                                    # 下载最近多少天的目标周期数据

    # ----- DeepSeek API 配置 -----
    DEEPSEEK_API_KEY = ""  # 请替换为你的真实密钥

    # ----- AI分析参数 -----
    LOOKBACK = 30                               # 发送最近多少根K线给AI
    # 提示词模板（{lookback}、{rule}、{data_text} 会被自动替换）
    PROMPT_TEMPLATE = """
你是一位专业的加密货币技术分析师。以下是 {symbol} 永续合约最近 {lookback} 根 {rule} K 线的数据（含常用技术指标）：

{data_text}

请根据这些数据撰写一份详细的技术分析报告，内容包括：
1. 整体趋势判断（上升/下降/震荡）
2. 关键的支撑位和阻力位
3. 成交量与价格的关系分析
4. RSI、MACD、KDJ 指标信号解读
5. 对后续走势的合理推测
6. 给出具体的交易建议：在什么位置做多，什么位置做空，什么时候买入/卖出，仓位部署建议（2倍杠杆，1:3备用金）。

"""

    # ----- 策略参数（用于指标计算）-----
    RSI_PERIOD = 14  # RSI计算周期，默认14，用于判断超买超卖
    MA_PERIOD = 10  # 移动平均线周期，用于计算MA10，观察中期趋势
    MA_FAST = 5  # 快速均线周期，用于计算MA5，反映短期趋势
    MA_SLOW = 30  # 慢速均线周期，用于计算MA30，反映中期趋势

    # ----- API 密钥（公共K线接口不需要）-----
    OKX_API_KEY = os.getenv("OKX_API_KEY", "")
    OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
    OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")
    SIMULATED = ""                             # 模拟盘标识

# ==================== 日志配置 ====================
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(Config.OUTPUT_DIR, "process.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== OKX API 客户端 ====================
class OKXClient:
    BASE_URL = "https://www.okx.com"

    def __init__(self, config: Config):
        self.config = config
        self.max_retries = 5
        self.retry_delay = 2
        self.headers = {}
        if config.SIMULATED:
            self.headers["x-simulated-trading"] = config.SIMULATED

    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        url = self.BASE_URL + endpoint
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"请求URL: {url}, 参数: {params}")
                resp = requests.get(url, params=params, headers=self.headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != "0":
                    raise Exception(f"API错误: {data.get('msg')} (code: {data.get('code')})")
                return data
            except Exception as e:
                logger.warning(f"请求失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error("达到最大重试次数，放弃请求。")
                    raise

    def get_klines(self, instId: str, bar: str, limit: int = 300, after: Optional[str] = None) -> List[List]:
        endpoint = "/api/v5/market/candles"
        params = {"instId": instId, "bar": bar, "limit": limit}
        if after:
            params["after"] = after
        data = self._request("GET", endpoint, params)
        return data.get("data", [])

# ==================== 数据加载与重采样 ====================
def load_history_data(history_dir: str, symbol: str) -> pd.DataFrame:
    """读取历史文件夹中所有CSV，合并为一个DataFrame"""
    all_files = [f for f in os.listdir(history_dir) if f.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"在 {history_dir} 中未找到CSV文件")
    df_list = []
    for file in all_files:
        filepath = os.path.join(history_dir, file)
        try:
            df = pd.read_csv(filepath)
            # 确保列名正确（与你提供的历史文件格式一致）
            required_cols = ['instrument_name', 'open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_quote', 'open_time', 'confirm']
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"文件 {file} 列名不匹配，跳过")
                continue
            # 只保留指定交易对的数据
            df = df[df['instrument_name'] == symbol]
            df_list.append(df)
        except Exception as e:
            logger.warning(f"读取文件 {file} 失败: {e}")
    if not df_list:
        raise ValueError(f"未能从历史数据中读取到交易对 {symbol} 的数据")
    combined = pd.concat(df_list, ignore_index=True)
    # 去重并排序
    combined['open_time'] = pd.to_datetime(combined['open_time'])
    combined = combined.drop_duplicates(subset=['open_time']).sort_values('open_time').reset_index(drop=True)
    logger.info(f"历史数据加载完成，总条数: {len(combined)}，时间范围: {combined['open_time'].min()} 至 {combined['open_time'].max()}")
    return combined

def resample_ohlcv(df: pd.DataFrame, target_freq: str, symbol: str) -> pd.DataFrame:
    """
    将1分钟K线重采样为目标周期
    target_freq: 例如 '15min', '1H', '1D' 等（pandas频率字符串）
    返回的列与原始文件一致：instrument_name, open, high, low, close, vol, vol_ccy, vol_quote, open_time, confirm
    """
    # 设置时间为索引
    df = df.set_index('open_time')
    # 确保数值列为float
    num_cols = ['open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_quote']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 定义重采样规则
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'vol_ccy': 'sum',
        'vol_quote': 'sum'
    }
    # 重采样
    resampled = df.resample(target_freq).agg(ohlc_dict).dropna(how='all')
    # 重置索引，open_time变为列
    resampled = resampled.reset_index()
    # 添加instrument_name列
    resampled['instrument_name'] = symbol
    # 添加confirm列（全部设为1，因为重采样后的K线视为完整）
    resampled['confirm'] = 1
    # 调整列顺序
    column_order = ['instrument_name', 'open', 'high', 'low', 'close', 'vol', 'vol_ccy', 'vol_quote', 'open_time', 'confirm']
    resampled = resampled[column_order]
    logger.info(f"重采样完成，生成 {len(resampled)} 条 {target_freq} K线")
    return resampled

# ==================== 下载最新数据 ====================
def fetch_target_klines(config: Config) -> pd.DataFrame:
    """下载目标周期的最新数据（基于时间范围）"""
    client = OKXClient(config)
    bar = config.TARGET_TIMEFRAME
    # 将目标周期转换为OKX API接受的bar格式（例如 '15m' -> '15m', '1H' -> '1H'）
    okx_bar = bar.replace('min', 'm')
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - config.DAYS * 24 * 60 * 60 * 1000

    all_data = []
    after_ts = None
    logger.info(f"开始下载 {config.SYMBOL} {okx_bar} 数据，时间范围: {datetime.fromtimestamp(start_ts/1000)} 至 {datetime.fromtimestamp(end_ts/1000)}")

    while True:
        batch = client.get_klines(config.SYMBOL, okx_bar, limit=300, after=after_ts)
        if not batch:
            break
        batch_oldest_ts = int(batch[-1][0])
        all_data.extend(batch)
        if batch_oldest_ts <= start_ts:
            break
        after_ts = batch[-1][0]
        time.sleep(0.2)

    if not all_data:
        raise Exception("未能获取任何新K线数据")

    # 转换为DataFrame，列名与历史文件保持一致
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close",
        "vol", "vol_ccy", "vol_quote", "confirm"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms")
    for col in ["open", "high", "low", "close", "vol", "vol_ccy", "vol_quote"]:
        df[col] = pd.to_numeric(df[col])

    df = df.sort_values("open_time").reset_index(drop=True)
    df = df[df["open_time"] >= pd.to_datetime(start_ts, unit="ms")].reset_index(drop=True)
    # 添加instrument_name列
    df.insert(0, "instrument_name", config.SYMBOL)
    logger.info(f"下载到 {len(df)} 条新K线数据")
    return df

# ==================== 合并数据 ====================
def merge_data(hist_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """合并历史重采样数据和新下载的数据，按open_time去重（保留新数据）"""
    combined = pd.concat([hist_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["open_time"], keep="last")
    combined = combined.sort_values("open_time").reset_index(drop=True)
    logger.info(f"合并后总数据量: {len(combined)} 条")
    return combined

# ==================== 计算技术指标 ====================
def calculate_indicators(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """计算MA, RSI, MACD, KDJ指标"""
    df = df.copy()
    df.set_index('open_time', inplace=True)
    # 移动平均线
    df['MA5'] = df['close'].rolling(window=config.MA_FAST).mean()
    df['MA10'] = df['close'].rolling(window=config.MA_PERIOD).mean()
    df['MA30'] = df['close'].rolling(window=config.MA_SLOW).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=config.RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=config.RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    # KDJ
    low_min = df['low'].rolling(window=9).min()
    high_max = df['high'].rolling(window=9).max()
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3 * k - 2 * d
    df['K'] = k
    df['D'] = d
    df['J'] = j

    df.reset_index(inplace=True)
    return df

# ==================== 仓位计算器 ====================
def position_calculator():
    """交互式仓位计算器（支持1:3资金管理策略、最小开仓数量和步长）"""
    print("\n🔢 仓位计算器启动（输入 q 随时退出）")
    try:
        # 输入账户余额
        balance = input("请输入账户总余额 (USDT): ").strip()
        if balance.lower() == 'q': return
        balance = float(balance)

        # 是否采用1:3资金管理策略
        use_1_3 = input("是否采用'1份仓位，3份备用金'策略？(y/n，默认n): ").strip().lower()
        if use_1_3 == 'q': return
        if use_1_3 in ['y', 'yes']:
            available_capital = balance / 4.0
            strategy_note = "（已启用1:3策略，实际可用资金 = 总余额/4）"
        else:
            available_capital = balance
            strategy_note = "（未启用1:3策略，全仓可用）"

        # 输入风险比例（基于总余额）
        risk_pct = input("请输入风险比例 (基于总余额的百分比，例如2表示2%): ").strip()
        if risk_pct.lower() == 'q': return
        risk_pct = float(risk_pct) / 100
        risk_amount = balance * risk_pct  # 基于总余额的风险金额

        # 输入入场价格
        entry = input("请输入入场价格 (USDT): ").strip()
        if entry.lower() == 'q': return
        entry = float(entry)

        # 输入止损价格
        stop = input("请输入止损价格 (USDT): ").strip()
        if stop.lower() == 'q': return
        stop = float(stop)

        # 输入合约乘数
        multiplier_input = input("请输入合约乘数 (每张合约的币数量，默认1): ").strip()
        if multiplier_input.lower() == 'q': return
        multiplier = float(multiplier_input) if multiplier_input else 1.0

        # 输入杠杆
        leverage_input = input("请输入杠杆倍数 (默认2): ").strip()
        if leverage_input.lower() == 'q': return
        leverage = float(leverage_input) if leverage_input else 2.0

        # 输入最小开仓数量（张）和步长
        min_contracts_input = input("请输入最小开仓数量 (张，如无要求请输入0): ").strip()
        if min_contracts_input.lower() == 'q': return
        min_contracts = float(min_contracts_input) if min_contracts_input else 0.0

        step_input = input("请输入开仓数量步长 (例如1表示整数张，0.001表示可精确到0.001张，默认1): ").strip()
        if step_input.lower() == 'q': return
        step = float(step_input) if step_input else 1.0
        if step <= 0:
            print("❌ 步长必须大于0，使用默认值1")
            step = 1.0

        # 计算止损距离
        stop_distance = abs(entry - stop)
        if stop_distance == 0:
            print("❌ 止损价格不能等于入场价格")
            return

        # 每张合约的亏损 = 止损距离 * 合约乘数
        loss_per_contract = stop_distance * multiplier

        # 基于风险金额的理论合约数
        theoretical_contracts = risk_amount / loss_per_contract

        # 根据步长调整合约数（向上取整）
        if step > 0:
            adjusted_contracts = math.ceil(theoretical_contracts / step) * step
        else:
            adjusted_contracts = theoretical_contracts

        # 确保满足最小开仓数量
        if min_contracts > 0 and adjusted_contracts < min_contracts:
            adjusted_contracts = min_contracts
            contract_note = f"⚠️ 理论合约数 {theoretical_contracts:.4f} 小于最小要求，已强制设为最小数量 {min_contracts}"
        else:
            contract_note = f"已根据步长 {step} 向上取整"

        # 重新计算实际风险金额和所需保证金
        actual_risk = adjusted_contracts * loss_per_contract
        notional_per_contract = entry * multiplier
        total_notional = adjusted_contracts * notional_per_contract
        margin = total_notional / leverage

        # 检查保证金是否超出可用资金
        margin_check = "✅ 保证金充足" if margin <= available_capital else f"⚠️ 保证金不足！需要 {margin:.2f} USDT，但可用资金只有 {available_capital:.2f} USDT。建议降低杠杆或减少仓位。"

        # 风险对比
        risk_diff = actual_risk - risk_amount
        if abs(risk_diff) < 0.01:
            risk_note = "实际风险金额与设定一致。"
        elif risk_diff > 0:
            risk_note = f"⚠️ 实际风险金额比设定值高出 {risk_diff:.2f} USDT ({(risk_diff/risk_amount*100):.1f}%)，请确认是否接受。"
        else:
            risk_note = f"✅ 实际风险金额比设定值低 {abs(risk_diff):.2f} USDT，风险更小。"

        # 输出结果
        result = f"""
📊 仓位计算结果 {strategy_note}
• 账户总余额: {balance:.2f} USDT
• 可用开仓资金: {available_capital:.2f} USDT
• 风险比例: {risk_pct*100:.2f}% → 设定风险金额: {risk_amount:.2f} USDT
• 入场价格: {entry:.4f} USDT
• 止损价格: {stop:.4f} USDT → 止损距离: {stop_distance:.4f} USDT
• 合约乘数: {multiplier} 币/张
• 杠杆倍数: {leverage}x
• 开仓数量约束: 最小 {min_contracts} 张, 步长 {step}

✅ 建议开仓数量: {adjusted_contracts:.4f} 张
💼 所需保证金: {margin:.2f} USDT
{margin_check}
⚠️ 实际风险金额: {actual_risk:.2f} USDT {risk_note}
"""
        print(result)
        return result

    except ValueError:
        print("❌ 输入无效，请输入数字")
        return None
    except KeyboardInterrupt:
        print("\n❌ 计算已取消")
        return None

# ==================== AI 分析模块 ====================
def ai_analysis(config: Config, df: pd.DataFrame) -> str:
    """调用DeepSeek API生成技术分析报告"""
    # 检查API密钥
    if not config.DEEPSEEK_API_KEY:
        logger.error("未设置DeepSeek API密钥")
        sys.exit(1)

    # 初始化客户端
    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com"
    )

    # 取最近LOOKBACK条数据，去除NaN
    recent = df.tail(config.LOOKBACK).dropna().round(4)
    if recent.empty:
        logger.error("数据不足，无法生成分析")
        return ""

    # 格式化数据为文本（只保留需要的列）
    cols_for_ai = ['open_time', 'open', 'high', 'low', 'close', 'vol',
                   'MA5', 'MA10', 'MA30', 'RSI14', 'MACD', 'Signal', 'MACD_Hist', 'K', 'D', 'J']
    # 确保列存在
    available_cols = [c for c in cols_for_ai if c in recent.columns]
    data_text = recent[available_cols].to_string(index=False)

    # 构建提示词
    prompt = config.PROMPT_TEMPLATE.format(
        symbol=config.SYMBOL,
        lookback=len(recent),
        rule=config.TARGET_TIMEFRAME,
        data_text=data_text
    )

    logger.info("正在请求DeepSeek API进行初次分析...")
    messages = [
        {"role": "system", "content": "你是一名经验丰富的交易员和技术分析师。"},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.5,
            max_tokens=2000,
            stream=False
        )
    except Exception as e:
        logger.error(f"API调用失败: {e}")
        return ""

    analysis = response.choices[0].message.content
    return analysis

def interactive_chat(config: Config, messages: List[Dict]) -> List[str]:
    """多轮交互对话，返回对话记录列表"""
    print("\n💬 现在您可以继续提问，结合最新市场情况进行更深入的分析。")
    print("🔄 输入“计算仓位”或“#calc”启动仓位计算器，输入“exit”结束对话。")

    chat_log = []  # 用于保存对话记录

    while True:
        user_input = input("\n👤 您: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("👋 对话结束。")
            break
        if not user_input:
            continue

        # 检测是否触发仓位计算器
        if user_input.lower() in ['计算仓位', '#calc', '/calc', '仓位计算']:
            calc_result = position_calculator()
            if calc_result:
                chat_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 用户启动仓位计算器")
                chat_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 系统: {calc_result}")
            continue

        # 记录用户提问
        chat_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 用户: {user_input}")

        # 将用户问题添加到对话历史
        messages.append({"role": "user", "content": user_input})

        # 调用API
        print("🤖 AI 思考中...")
        try:
            client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.5,
                max_tokens=2000,
                stream=False
            )
        except Exception as e:
            print(f"❌ API调用失败：{e}")
            break

        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})

        # 记录AI回复
        chat_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] AI: {reply}")

        print("\n🤖 AI: " + reply)
        print("-" * 50)

    return chat_log

# ==================== 主程序 ====================
def main():
    config = Config()
    # 交互式输入交易对
    if config.SYMBOL is None:
        config.SYMBOL = input("请输入交易对（例如 ESP-USDT-SWAP）: ").strip()

    # 生成累积文件名
    merged_filename = f"{config.SYMBOL}_{config.TARGET_TIMEFRAME}_all.csv"
    merged_path = os.path.join(config.OUTPUT_DIR, merged_filename)

    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    try:
        # 1. 加载历史1分钟数据
        logger.info("步骤1：加载历史数据...")
        hist_raw = load_history_data(config.HISTORY_DIR, config.SYMBOL)

        # 2. 重采样为目标周期
        logger.info(f"步骤2：将历史数据重采样为 {config.TARGET_TIMEFRAME}...")
        # 将目标周期转换为pandas频率字符串
        freq_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1H': '1H', '2H': '2H', '4H': '4H', '6H': '6H', '12H': '12H',
            '1D': '1D', '1W': '1W', '1M': '1M'
        }
        if config.TARGET_TIMEFRAME not in freq_map:
            raise ValueError(f"不支持的目标周期: {config.TARGET_TIMEFRAME}")
        target_freq = freq_map[config.TARGET_TIMEFRAME]
        hist_resampled = resample_ohlcv(hist_raw, target_freq, config.SYMBOL)

        # 3. 下载最新目标周期数据
        logger.info("步骤3：下载最新数据...")
        new_data = fetch_target_klines(config)

        # 4. 合并数据
        logger.info("步骤4：合并数据...")
        merged = merge_data(hist_resampled, new_data)

        # 5. 计算技术指标
        logger.info("步骤5：计算技术指标...")
        df_with_indicators = calculate_indicators(merged, config)

        # 6. 保存合并后的累积文件（含指标）
        df_with_indicators.to_csv(merged_path, index=False, encoding='utf-8-sig')
        logger.info(f"合并数据已保存至: {merged_path}")

        # 7. AI分析
        logger.info("步骤6：调用DeepSeek API生成分析报告...")
        analysis = ai_analysis(config, df_with_indicators)
        if analysis:
            # 保存分析报告
            now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(config.OUTPUT_DIR, f"技术报告_{now_str}.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(analysis)
            print(f"\n✅ 分析报告已保存至：{report_path}")
            print("\n" + "="*60)
            print("📊 AI 技术分析报告")
            print("="*60)
            print(analysis)
            print("="*60)

            # 初始化对话历史（包含本次分析）
            messages = [
                {"role": "system", "content": "你是一名经验丰富的交易员和技术分析师。"},
                {"role": "assistant", "content": analysis}
            ]

            # 8. 交互对话
            chat_log = interactive_chat(config, messages)

            # 9. 保存对话记录
            if chat_log:
                chat_path = os.path.join(config.OUTPUT_DIR, f"对话_{now_str}.txt")
                with open(chat_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(chat_log))
                print(f"✅ 对话记录已保存至：{chat_path}")

        logger.info("全部处理完成！")

    except Exception as e:
        logger.error(f"运行失败: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()