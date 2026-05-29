"""
22_live_trader_gbpjpy.py
=========================
Live торговый скрипт стратегии GBP/JPY RSI fade short для MT5 demo.

ЛОГИКА:
1. Запускается по расписанию (например каждые 15 минут на закрытии M15-бара)
2. Подключается к MT5
3. ПРОВЕРКИ БЕЗОПАСНОСТИ (если что-то не так — выход без действий):
   - Это demo-счёт?
   - Файл STOP.txt существует? (kill switch)
   - Открыты ли уже наши позиции? (только одна за раз)
   - Дневная просадка не превышена?
4. Управление существующей позицией:
   - Если есть позиция и держится > MAX_HOLD_BARS — закрываем (таймаут)
5. Поиск нового сигнала:
   - Качаем последние 100 баров M15
   - Считаем RSI(14) на закрытии последнего ЗАКРЫТОГО бара
   - Если RSI > RSI_LEVEL И час во FLAT_HOURS → отправляем SELL ордер
6. Логирование всего происходящего

ПАРАМЕТРЫ СТРАТЕГИИ (фиксированные после бэктеста 21_*):
   RSI > 80, SL = 2.0 × ATR, TP = 2.0 × ATR, hold ≤ 16 баров

Запуск:
    python 22_live_trader_gbpjpy.py
    
Для автозапуска по расписанию — настроить Task Scheduler Windows
на запуск этого скрипта каждые 15 минут.

ОСТАНОВКА: создать файл STOP.txt в папке проекта.
"""

import os
import sys
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: пакет MetaTrader5 не установлен.")
    sys.exit(1)


# ============== КОНФИГУРАЦИЯ СТРАТЕГИИ ==============
SYMBOL_CANDIDATES = ["GBPJPY", "GBP/JPY", "GBPJPYm", "GBPJPY."]
LOT_SIZE          = 0.01           # ТОЛЬКО минимальный лот
PIP_SIZE          = 0.01           # JPY-пара
COST_PIPS         = 1.7            # для расчёта margin requirements

RSI_PERIOD        = 14
RSI_LEVEL         = 80             # RSI > 80 = сигнал
FLAT_HOURS        = [0, 1, 2, 5, 6, 7, 22, 23]
SL_MULT           = 2.0            # SL = entry + ATR × 2.0
TP_MULT           = 2.0            # TP = entry - ATR × 2.0
ATR_PERIOD        = 14
MAX_HOLD_BARS     = 16             # 16 × 15 мин = 4 часа

TIMEFRAME         = mt5.TIMEFRAME_M15
BARS_TO_DOWNLOAD  = 100            # для расчёта индикаторов

# ============== ЗАЩИТЫ ==============
ONLY_DEMO         = True           # ЖЁСТКО: только demo, иначе выход
MAX_DAILY_LOSS_PCT = 1.0           # kill switch на 1% дневной просадки
STOP_FILE         = "./STOP.txt"   # kill switch файл
STATE_FILE        = "./live_state.json"
LOG_FILE          = "./live_trader.log"

MAGIC_NUMBER      = 20260529       # уникальный ID — чтобы трогать только свои сделки


# ============== ЛОГИРОВАНИЕ ==============
def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{level}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============== STATE ==============
def load_state():
    if not os.path.exists(STATE_FILE):
        return {"daily_start_equity": None, "daily_date": None, "last_signal_time": None}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"daily_start_equity": None, "daily_date": None, "last_signal_time": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ============== ИНДИКАТОРЫ ==============
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, period=14):
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ============== ПРОВЕРКИ БЕЗОПАСНОСТИ ==============
def safety_check_stop_file():
    if os.path.exists(STOP_FILE):
        log(f"KILL SWITCH: найден файл {STOP_FILE}. Выход.", "WARN")
        return False
    return True


def safety_check_demo(acc):
    is_demo = "demo" in (acc.server or "").lower()
    if ONLY_DEMO and not is_demo:
        log(f"FATAL: счёт НЕ demo! Сервер: {acc.server}. Скрипт защищён, не торгует на REAL.", "ERROR")
        return False
    return True


def safety_check_daily_loss(acc, state):
    today = datetime.now().date().isoformat()
    if state.get("daily_date") != today:
        # Новый день — обнуляем
        state["daily_date"] = today
        state["daily_start_equity"] = acc.equity
        save_state(state)
        log(f"Новый торговый день. Стартовый equity: {acc.equity}")
        return True

    start_eq = state["daily_start_equity"]
    if start_eq is None or start_eq <= 0:
        return True

    daily_pnl_pct = (acc.equity - start_eq) / start_eq * 100
    if daily_pnl_pct < -MAX_DAILY_LOSS_PCT:
        log(f"KILL SWITCH: дневная просадка {daily_pnl_pct:.2f}% превысила лимит "
            f"-{MAX_DAILY_LOSS_PCT}%. Закрываю все наши позиции, выхожу.", "WARN")
        close_all_our_positions()
        return False
    return True


# ============== РАБОТА С ПОЗИЦИЯМИ ==============
def get_our_positions(symbol):
    """Возвращает только позиции с нашим magic number по символу."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []
    return [p for p in positions if p.magic == MAGIC_NUMBER]


def close_position(position):
    """Закрывает позицию по market."""
    symbol = position.symbol
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log(f"Не получен tick для {symbol}, не могу закрыть", "ERROR")
        return False

    if position.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    request = {
        "action":   mt5.TRADE_ACTION_DEAL,
        "symbol":   symbol,
        "volume":   position.volume,
        "type":     order_type,
        "position": position.ticket,
        "price":    price,
        "deviation": 20,
        "magic":    MAGIC_NUMBER,
        "comment":  "close by script",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"Ошибка закрытия позиции {position.ticket}: "
            f"{result.retcode if result else 'None'} - "
            f"{result.comment if result else 'No result'}", "ERROR")
        return False
    log(f"Позиция {position.ticket} закрыта по {price}")
    return True


def close_all_our_positions():
    positions = mt5.positions_get()
    if positions is None:
        return
    for p in positions:
        if p.magic == MAGIC_NUMBER:
            close_position(p)


def check_position_timeout(position, max_bars):
    """Закрывает позицию если держится дольше max_bars."""
    open_time = datetime.fromtimestamp(position.time)
    now = datetime.now()
    held_minutes = (now - open_time).total_seconds() / 60
    max_minutes = max_bars * 15  # M15
    if held_minutes > max_minutes:
        log(f"Позиция {position.ticket} держится {held_minutes:.0f} мин > "
            f"{max_minutes} мин лимита. Принудительно закрываю.", "WARN")
        close_position(position)
        return True
    return False


# ============== ТОРГОВАЯ ЛОГИКА ==============
def find_symbol():
    for c in SYMBOL_CANDIDATES:
        info = mt5.symbol_info(c)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(c, True)
            return c
    return None


def check_signal(symbol):
    """Качает последние бары и проверяет условие сигнала."""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS_TO_DOWNLOAD)
    if rates is None or len(rates) < ATR_PERIOD + RSI_PERIOD + 5:
        log(f"Не удалось получить бары для {symbol}", "ERROR")
        return None

    df = pd.DataFrame(rates)
    df["timestamps"] = pd.to_datetime(df["time"], unit="s")
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["atr"] = atr(df, ATR_PERIOD)

    # ВАЖНО: смотрим на последний ЗАКРЫТЫЙ бар (а не текущий формируемый).
    # copy_rates_from_pos возвращает массив, где [0] — самый старый, [-1] — последний.
    # Последний бар может быть ещё формирующимся. Берём предпоследний для надёжности.
    last_closed_idx = -2  # предпоследний бар (точно закрытый)

    last_bar = df.iloc[last_closed_idx]
    last_rsi = last_bar["rsi"]
    last_atr = last_bar["atr"]
    last_close = last_bar["close"]
    last_time = last_bar["timestamps"]
    last_hour = last_time.hour

    log(f"Последний закрытый M15-бар: {last_time}, close={last_close:.3f}, "
        f"RSI={last_rsi:.1f}, ATR={last_atr:.3f}, hour={last_hour}")

    # Условие сигнала
    signal = (last_rsi > RSI_LEVEL) and (last_hour in FLAT_HOURS)
    if signal:
        log(f"СИГНАЛ: RSI={last_rsi:.1f} > {RSI_LEVEL} и час {last_hour} ∈ FLAT_HOURS", "SIGNAL")
        return {
            "atr": last_atr,
            "last_close": last_close,
            "last_time": last_time,
        }
    return None


def place_short_order(symbol, atr_value):
    """Размещает SELL ордер с SL и TP."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log("Не получен tick", "ERROR")
        return False

    info = mt5.symbol_info(symbol)
    entry_price = tick.bid  # для SELL входим по bid
    sl_price = entry_price + atr_value * SL_MULT
    tp_price = entry_price - atr_value * TP_MULT

    # Округление до digits
    sl_price = round(sl_price, info.digits)
    tp_price = round(tp_price, info.digits)

    # Проверка stops level
    min_dist = info.trade_stops_level * info.point
    if abs(sl_price - entry_price) < min_dist or abs(tp_price - entry_price) < min_dist:
        log(f"SL/TP слишком близко к цене (мин {min_dist}). Пропускаю сделку.", "WARN")
        return False

    request = {
        "action":   mt5.TRADE_ACTION_DEAL,
        "symbol":   symbol,
        "volume":   LOT_SIZE,
        "type":     mt5.ORDER_TYPE_SELL,
        "price":    entry_price,
        "sl":       sl_price,
        "tp":       tp_price,
        "deviation": 20,
        "magic":    MAGIC_NUMBER,
        "comment":  f"RSI>{RSI_LEVEL} fade short",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    log(f"Отправляю SELL ордер: {LOT_SIZE} лот @ {entry_price}, "
        f"SL={sl_price}, TP={tp_price}", "ORDER")
    result = mt5.order_send(request)
    if result is None:
        log(f"order_send вернул None: {mt5.last_error()}", "ERROR")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"Ордер отклонён: retcode={result.retcode}, comment={result.comment}", "ERROR")
        return False

    log(f"ОРДЕР ИСПОЛНЕН: ticket={result.order}, цена={result.price}", "ORDER")
    return True


# ============== ГЛАВНАЯ ЛОГИКА ==============
def main():
    log("=" * 60)
    log("Запуск live торгового скрипта")

    # 1. STOP file
    if not safety_check_stop_file():
        sys.exit(0)

    # 2. Подключение к MT5
    if not mt5.initialize():
        log(f"MT5 не подключился: {mt5.last_error()}", "ERROR")
        sys.exit(1)

    acc = mt5.account_info()
    if acc is None:
        log("Не получен account_info — не залогинен?", "ERROR")
        mt5.shutdown()
        sys.exit(1)

    log(f"Подключён: {acc.login} ({acc.server}), баланс={acc.balance} {acc.currency}, "
        f"equity={acc.equity}")

    # 3. Проверка demo
    if not safety_check_demo(acc):
        mt5.shutdown()
        sys.exit(1)

    # 4. Проверка дневной просадки (kill switch)
    state = load_state()
    if not safety_check_daily_loss(acc, state):
        mt5.shutdown()
        sys.exit(0)

    # 5. Поиск символа
    symbol = find_symbol()
    if symbol is None:
        log("GBPJPY не найден у брокера", "ERROR")
        mt5.shutdown()
        sys.exit(1)
    log(f"Торговый символ: {symbol}")

    # 6. Проверка существующих позиций — таймаут
    our_positions = get_our_positions(symbol)
    log(f"Наших открытых позиций: {len(our_positions)}")
    for p in our_positions:
        if check_position_timeout(p, MAX_HOLD_BARS):
            our_positions = get_our_positions(symbol)  # обновляем список

    # 7. Если уже есть позиция — не открываем новую
    if len(our_positions) > 0:
        log(f"Уже в позиции {our_positions[0].ticket}, новых сделок не открываю")
        mt5.shutdown()
        sys.exit(0)

    # 8. Поиск сигнала
    signal_info = check_signal(symbol)
    if signal_info is None:
        log("Сигнала нет")
        mt5.shutdown()
        sys.exit(0)

    # 9. Защита от дублирования: не открываем второй раз на тот же бар
    last_signal_time = state.get("last_signal_time")
    signal_time_str = signal_info["last_time"].isoformat()
    if last_signal_time == signal_time_str:
        log("Этот бар уже обработан, не дублирую сигнал")
        mt5.shutdown()
        sys.exit(0)

    # 10. Размещение ордера
    success = place_short_order(symbol, signal_info["atr"])
    if success:
        state["last_signal_time"] = signal_time_str
        save_state(state)

    mt5.shutdown()
    log("Цикл завершён")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"НЕОБРАБОТАННАЯ ОШИБКА: {e}", "ERROR")
        import traceback
        log(traceback.format_exc(), "ERROR")
        sys.exit(1)