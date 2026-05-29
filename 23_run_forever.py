"""
23_run_forever.py
==================
Запускает 22_live_trader_gbpjpy.py каждые 15 минут в бесконечном цикле.

ОСОБЕННОСТИ:
- Запуск ВЫРАВНЕН по закрытию M15-бара: HH:01, HH:16, HH:31, HH:46
  (через 1 минуту после закрытия, чтобы бар точно был доступен)
- Ловит ошибки исполнения торгового скрипта и продолжает работать
- При создании файла STOP.txt — останавливается
- Ctrl+C — корректное завершение

ИСПОЛЬЗОВАНИЕ:
    python 23_run_forever.py

Чтобы остановить:
    1. Создать файл STOP.txt в этой папке (мягкая остановка после текущего цикла)
    2. Или Ctrl+C в окне терминала
"""

import os
import sys
import subprocess
import time
from datetime import datetime, timedelta

TRADER_SCRIPT = "22_live_trader_gbpjpy.py"
STOP_FILE     = "./STOP.txt"
WRAPPER_LOG   = "./run_forever.log"

# Запуск ВЫРАВНЕН по 15-минутным барам + 60 секунд буфера для закрытия
EXECUTE_OFFSET_SECONDS = 60   # запускаем через 60 сек после HH:00, HH:15, HH:30, HH:45


def wlog(msg):
    """Логирование обёртки (отдельно от логов трейдера)."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [WRAPPER] {msg}"
    print(line, flush=True)
    try:
        with open(WRAPPER_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def time_until_next_run():
    """Сколько секунд до следующего HH:01, HH:16, HH:31, HH:46."""
    now = datetime.now()
    # Ближайший целевой момент: ближайший XX:M:00 где M ∈ {1, 16, 31, 46}
    minute = now.minute
    second = now.second

    # Ближайший XX:00, XX:15, XX:30, XX:45
    next_quarter_minute = ((minute // 15) + 1) * 15
    if next_quarter_minute >= 60:
        next_quarter_minute = 0
        target_hour = (now.hour + 1) % 24
        target = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        if target_hour == 0:
            target = target + timedelta(days=1)
    else:
        target = now.replace(minute=next_quarter_minute, second=0, microsecond=0)

    # Прибавляем буфер
    target = target + timedelta(seconds=EXECUTE_OFFSET_SECONDS)

    delta = (target - datetime.now()).total_seconds()
    if delta < 0:
        # Если уже прошли — следующий через 15 минут
        delta += 15 * 60
    return delta


def run_trader_once():
    """Запускает торговый скрипт как subprocess. Возвращает exit code."""
    if not os.path.exists(TRADER_SCRIPT):
        wlog(f"ERROR: {TRADER_SCRIPT} не найден")
        return -1

    wlog(f"Запускаю {TRADER_SCRIPT}...")
    try:
        result = subprocess.run(
            [sys.executable, TRADER_SCRIPT],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,  # максимум 2 мин на один цикл
        )
        # Пишем stdout трейдера в обёрточный лог (опционально)
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print("  > " + line, flush=True)
        if result.returncode != 0:
            wlog(f"Трейдер завершился с кодом {result.returncode}")
            if result.stderr:
                wlog(f"STDERR: {result.stderr[:500]}")
        else:
            wlog(f"Трейдер завершился успешно")
        return result.returncode
    except subprocess.TimeoutExpired:
        wlog(f"TIMEOUT: трейдер не завершился за 120 сек, убит")
        return -2
    except Exception as e:
        wlog(f"Ошибка запуска трейдера: {e}")
        return -3


def main():
    wlog("=" * 60)
    wlog("ЗАПУСК WRAPPER (постоянный режим)")
    wlog("Расписание: HH:01:00, HH:16:00, HH:31:00, HH:46:00")
    wlog(f"Чтобы остановить: создать {STOP_FILE} или Ctrl+C")
    wlog("=" * 60)

    cycle_count = 0
    try:
        while True:
            # Проверка STOP файла
            if os.path.exists(STOP_FILE):
                wlog(f"Найден {STOP_FILE} — мягкая остановка")
                break

            # Ждём до следующего запуска
            wait_sec = time_until_next_run()
            next_run_time = datetime.now() + timedelta(seconds=wait_sec)
            wlog(f"Жду {wait_sec:.0f} сек до следующего запуска "
                 f"({next_run_time.strftime('%H:%M:%S')})")

            # Прерываемое ожидание (чтобы можно было Ctrl+C прервать)
            sleep_start = time.time()
            while time.time() - sleep_start < wait_sec:
                if os.path.exists(STOP_FILE):
                    break
                time.sleep(min(5.0, wait_sec - (time.time() - sleep_start)))

            if os.path.exists(STOP_FILE):
                continue

            # Запуск
            cycle_count += 1
            wlog(f"--- Цикл #{cycle_count} ---")
            run_trader_once()

    except KeyboardInterrupt:
        wlog("Получен Ctrl+C — выход")
    except Exception as e:
        wlog(f"НЕОЖИДАННАЯ ОШИБКА в обёртке: {e}")
        import traceback
        wlog(traceback.format_exc())
    finally:
        wlog(f"Всего циклов: {cycle_count}")
        wlog("Wrapper остановлен")


if __name__ == "__main__":
    main()