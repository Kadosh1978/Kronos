"""
11_diagnose_fxpro.py
=====================
Диагностический скрипт: вытаскивает всю информацию из FxPro MT5 API
по EUR/USD и BTC/USD. Цель — понять полную стоимость сделки на M5.

Что делает:
1. Показывает информацию о подключённом счёте (тип, сервер, баланс)
2. Находит EUR/USD и BTC/USD с учётом возможных суффиксов
3. Печатает ВСЕ доступные поля symbol_info для каждого
4. Получает текущий tick и считает реальный спред в пипсах
5. Симулирует расчёт прибыли по сделке на 0.01 лота, +1 пипс движения
   — это покажет нам комиссию опосредованно (через сравнение
   "что должно было быть" и "что реально получаем")

Запуск:
    python 11_diagnose_fxpro.py

Перед запуском: MT5 терминал ЗАПУЩЕН и залогинен в FxPro demo.
"""

import sys
import MetaTrader5 as mt5


# Возможные имена символов у FxPro (с суффиксами и без)
SYMBOL_CANDIDATES = {
    "EURUSD": ["EURUSD", "EURUSD.", "EURUSD-", "EURUSDm", "EURUSD.r", "EURUSD.raw"],
    "BITCOIN": ["BITCOIN", "BTCUSD", "BTCUSD.", "BTC/USD", "BTCUSDm", "BTC"],
}


def init_mt5():
    if not mt5.initialize():
        print(f"ERROR: MT5 не подключился: {mt5.last_error()}")
        print("Убедись что MT5 терминал FxPro запущен и залогинен.")
        sys.exit(1)

    acc = mt5.account_info()
    term = mt5.terminal_info()

    print("=" * 70)
    print("ИНФОРМАЦИЯ О СЧЁТЕ И ТЕРМИНАЛЕ")
    print("=" * 70)
    print(f"Терминал:    {term.company if term else '?'}")
    if acc:
        is_demo = "demo" in (acc.server or "").lower()
        print(f"Счёт:        {acc.login}")
        print(f"Сервер:      {acc.server}")
        print(f"Тип:         {'DEMO' if is_demo else 'REAL'}")
        print(f"Валюта:      {acc.currency}")
        print(f"Баланс:      {acc.balance}")
        print(f"Equity:      {acc.equity}")
        print(f"Маржа:       {acc.margin}")
        print(f"Свободно:    {acc.margin_free}")
        print(f"Кредитное плечо: 1:{acc.leverage}")
        print(f"Тип маржинальности: {acc.margin_mode}")  # 0=retail-FX hedging, 1=netting, 2=exchange
    print("=" * 70)


def find_symbol(base, candidates):
    """Ищет правильное имя символа у текущего брокера."""
    # Сначала точные совпадения
    for c in candidates:
        info = mt5.symbol_info(c)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(c, True)
            return c

    # Потом подстрочный поиск (например "BTC" в любом имени)
    all_syms = mt5.symbols_get()
    matches = []
    base_short = base.replace("/", "").replace(".", "")
    for s in all_syms:
        if base_short in s.name.upper().replace("/", "").replace(".", ""):
            matches.append(s.name)
    if matches:
        print(f"  Не нашёл точное совпадение для {base}. Похожие в Market Watch:")
        for m in matches[:10]:
            print(f"    {m}")
        # Берём первый
        chosen = matches[0]
        info = mt5.symbol_info(chosen)
        if info and not info.visible:
            mt5.symbol_select(chosen, True)
        print(f"  Использую: {chosen}")
        return chosen

    return None


def dump_symbol_info(symbol):
    """Печатает все поля symbol_info."""
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"  Символ {symbol} не найден.")
        return

    print(f"\n{'=' * 70}")
    print(f"СИМВОЛ: {symbol}")
    print(f"{'=' * 70}")

    # Основное
    print(f"Описание:           {info.description}")
    print(f"Базовая валюта:     {info.currency_base}")
    print(f"Валюта прибыли:     {info.currency_profit}")
    print(f"Валюта маржи:       {info.currency_margin}")
    print(f"Digits:             {info.digits}")
    print(f"Point:              {info.point}")

    # Спред
    print(f"\nСПРЕД И ИЗДЕРЖКИ:")
    print(f"  Spread (points):      {info.spread}")
    pip_size = info.point * 10 if info.digits in (3, 5) else info.point
    spread_pips = info.spread * info.point / pip_size
    print(f"  Spread (пипсы):       {spread_pips:.2f}")
    print(f"  Pip size (расчётный): {pip_size}")
    print(f"  Spread float:         {info.spread_float}  (True = плавающий)")

    # Лот
    print(f"\nОБЪЁМ:")
    print(f"  Контракт:           {info.trade_contract_size}")
    print(f"  Мин лот:            {info.volume_min}")
    print(f"  Макс лот:           {info.volume_max}")
    print(f"  Шаг лота:           {info.volume_step}")
    print(f"  Tick value:         {info.trade_tick_value}  (стоимость одного тика)")
    print(f"  Tick size:          {info.trade_tick_size}")

    # Стопы и торговые ограничения
    print(f"\nТОРГОВЫЕ ОГРАНИЧЕНИЯ:")
    print(f"  Stops level:        {info.trade_stops_level}  (мин расст стопа в points)")
    print(f"  Freeze level:       {info.trade_freeze_level}")
    print(f"  Trade mode:         {info.trade_mode}  (0=disabled, 4=full)")
    print(f"  Order mode:         {info.order_mode}")
    print(f"  Filling mode:       {info.filling_mode}  (1=FOK, 2=IOC)")

    # Свопы
    print(f"\nСВОПЫ:")
    print(f"  Swap long:          {info.swap_long}")
    print(f"  Swap short:         {info.swap_short}")
    print(f"  Swap mode:          {info.swap_mode}")
    print(f"  Тройной своп:       {info.swap_rollover3days}  (день недели 0-6)")

    # Маржа
    print(f"\nМАРЖА:")
    print(f"  Initial margin:     {info.margin_initial}")
    print(f"  Maintenance:        {info.margin_maintenance}")

    # Live tick
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"\nТЕКУЩИЙ TICK:")
        print(f"  Bid:                {tick.bid}")
        print(f"  Ask:                {tick.ask}")
        print(f"  Spread сейчас:      {tick.ask - tick.bid:.{info.digits}f}  "
              f"({(tick.ask - tick.bid) / pip_size:.2f} пипса)")
        print(f"  Time:               {tick.time}")

    # Расчёт реальной стоимости 0.01 лота
    print(f"\nСТОИМОСТЬ ОДНОЙ СДЕЛКИ 0.01 ЛОТА:")
    if tick:
        # Стоимость спреда (на вход = половина спреда, на выход = половина)
        cost_spread = (tick.ask - tick.bid) * info.trade_contract_size * 0.01
        print(f"  Спред round-trip:   {cost_spread:.4f} {info.currency_profit}")

        # 1 пипс на 0.01 лот
        pip_value_001 = pip_size * info.trade_contract_size * 0.01
        print(f"  Стоимость 1 пипса:  {pip_value_001:.4f} {info.currency_profit}")

        # Сколько пипсов = round-trip спред
        print(f"  Спред = {(tick.ask - tick.bid) / pip_size:.2f} пипсов в сторону")

    # Комиссия — пытаемся вычислить через order_calc_profit
    print(f"\nПРОВЕРКА КОМИССИИ (через симуляцию):")
    if tick:
        # Симулируем буй и продажу через 1 пипс
        profit_check = mt5.order_calc_profit(
            mt5.ORDER_TYPE_BUY, symbol, 0.01,
            tick.ask, tick.ask + pip_size
        )
        if profit_check is not None:
            print(f"  Расчётная прибыль (купил, +1 пипс, без комиссии):")
            print(f"    {profit_check:.4f} {info.currency_profit}")
            print(f"  Если в реальной сделке прибыль меньше — разница = комиссия.")
        else:
            print(f"  order_calc_profit вернул None: {mt5.last_error()}")

        margin_check = mt5.order_calc_margin(
            mt5.ORDER_TYPE_BUY, symbol, 0.01, tick.ask
        )
        if margin_check is not None:
            print(f"  Требуется маржи на 0.01 лот: {margin_check:.2f} {info.currency_margin}")


def main():
    init_mt5()

    print("\nИщу символы EUR/USD и BTC/USD...")
    found = {}
    for base, candidates in SYMBOL_CANDIDATES.items():
        sym = find_symbol(base, candidates)
        if sym:
            found[base] = sym
            print(f"  {base} -> {sym}")
        else:
            print(f"  {base} -> НЕ НАЙДЕН")

    if not found:
        print("\nНи один символ не найден. Открой Market Watch, добавь EUR/USD и BTC/USD.")
        mt5.shutdown()
        sys.exit(1)

    for base, sym in found.items():
        dump_symbol_info(sym)

    print("\n" + "=" * 70)
    print("ГОТОВО")
    print("=" * 70)
    print("\nЧто важно посмотреть:")
    print("  1. Spread (пипсы) — текущий и из symbol_info")
    print("  2. Stops level — минимальное расстояние стопа")
    print("  3. Filling mode — это влияет на тип ордеров в коде")
    print("  4. Маржа на 0.01 лот — сколько съедает одна сделка")
    print("  5. Комиссию ищи в Specification окне терминала вручную —")
    print("     через API она не отдаётся явно.")

    mt5.shutdown()


if __name__ == "__main__":
    main()