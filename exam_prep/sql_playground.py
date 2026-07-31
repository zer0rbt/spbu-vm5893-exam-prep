"""Песочница для тренировки SQL к экзамену.

Поднимает базу с двумя наборами данных и даёт SQL-консоль:

* `trades`  — история торгов из task_1.parquet / task_2.parquet;
* `bookings`, `tickets`, `ticket_flights` — синтетическая БД авиаперевозок
  по образцу из Раздела 2 программы (генерируется с фиксированным seed,
  поэтому результаты воспроизводимы).

Движки:
    duckdb  (по умолчанию) — почти полный SQL:2016: оконные функции, CTE,
             generate_series, QUALIFY, читает parquet напрямую;
    sqlite  — стандартная библиотека, ничего ставить не нужно; диалект беднее
             (нет generate_series как функции, нет FULL JOIN до 3.39,
             нет типа TIMESTAMP), полезен как проверка на переносимость.

Запуск:
    python sql_playground.py                       # интерактивная консоль
    python sql_playground.py -c "SELECT 1"         # одна команда
    python sql_playground.py -f queries.sql        # файл со скриптом
    python sql_playground.py --engine sqlite       # другой движок

В консоли:
    ;              завершает запрос (можно писать в несколько строк)
    \\dt            список таблиц
    \\d <table>     схема таблицы
    \\q             выход
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RNG_SEED = 42


# --------------------------------------------------------------------------- #
# Синтетическая БД авиаперевозок
# --------------------------------------------------------------------------- #
def make_airline_data(n_bookings: int = 5000) -> dict[str, pd.DataFrame]:
    """Генерирует bookings / tickets / ticket_flights с фиксированным seed.

    Суммы подобраны так, чтобы гистограмма с шагом 100 000 имела несколько
    непустых корзин и хотя бы одну пустую — иначе на задаче из программы
    не видно разницы между GROUP BY и LEFT JOIN по сетке корзин.
    """
    rng = np.random.default_rng(RNG_SEED)

    # Логнормальное распределение сумм: много дешёвых броней, длинный правый хвост.
    amounts = np.round(rng.lognormal(mean=11.2, sigma=0.9, size=n_bookings), 2)
    book_refs = [f"{i:06X}" for i in range(n_bookings)]
    book_dates = pd.Timestamp("2026-01-01") + pd.to_timedelta(
        rng.integers(0, 180 * 24 * 3600, size=n_bookings), unit="s"
    )

    bookings = pd.DataFrame(
        {"book_ref": book_refs, "book_date": book_dates, "total_amount": amounts}
    )

    # 1..3 билета на бронирование
    tickets_per_booking = rng.integers(1, 4, size=n_bookings)
    ticket_rows = []
    ticket_no = 1_000_000_000_000
    first_names = ["IVAN", "PETR", "ANNA", "OLGA", "SERGEY", "MARIA", "DMITRY", "ELENA"]
    last_names = ["IVANOV", "PETROV", "SIDOROV", "KUZNETSOV", "SMIRNOV", "POPOV"]
    for ref, k in zip(book_refs, tickets_per_booking):
        for _ in range(k):
            ticket_no += 1
            ticket_rows.append(
                {
                    "ticket_no": str(ticket_no),
                    "book_ref": ref,
                    "passenger_id": f"{rng.integers(1000, 9999)} {rng.integers(100000, 999999)}",
                    "passenger_name": f"{rng.choice(first_names)} {rng.choice(last_names)}",
                }
            )
    tickets = pd.DataFrame(ticket_rows)

    # 1..2 перелёта на билет; сумма перелётов примерно бьётся с total_amount
    flights_per_ticket = rng.integers(1, 3, size=len(tickets))
    fare = rng.choice(["Economy", "Comfort", "Business"], size=len(tickets),
                      p=[0.75, 0.15, 0.10])
    tf_rows = []
    for (_, t), k, f in zip(tickets.iterrows(), flights_per_ticket, fare):
        base = bookings.loc[bookings["book_ref"] == t["book_ref"], "total_amount"].iloc[0]
        for j in range(k):
            tf_rows.append(
                {
                    "ticket_no": t["ticket_no"],
                    "flight_id": int(rng.integers(1, 5000)),
                    "fare_conditions": f,
                    "amount": round(float(base) / (k * 2), 2),
                }
            )
    ticket_flights = pd.DataFrame(tf_rows)

    return {"bookings": bookings, "tickets": tickets, "ticket_flights": ticket_flights}


# --------------------------------------------------------------------------- #
# Подготовка соединения
# --------------------------------------------------------------------------- #
def open_duckdb(trades_path: Path | None):
    import duckdb

    con = duckdb.connect(":memory:")
    for name, df in make_airline_data().items():
        con.register(f"_{name}", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _{name}")
        con.unregister(f"_{name}")
    if trades_path is not None:
        con.execute(
            "CREATE TABLE trades AS "
            "SELECT *, price * quantity AS turnover FROM read_parquet(?)",
            [str(trades_path)],
        )
    return con


def open_sqlite(trades_path: Path | None):
    con = sqlite3.connect(":memory:")
    for name, df in make_airline_data().items():
        df.to_sql(name, con, index=False)
    if trades_path is not None:
        df = pd.read_parquet(trades_path)
        df["turnover"] = df["price"] * df["quantity"]
        # sqlite не знает datetime — храним ISO-строку, сравнения по ней работают.
        df["timestamp"] = df["timestamp"].astype(str)
        df.to_sql("trades", con, index=False, chunksize=100_000)
    return con


def run_sql(con, engine: str, sql: str) -> pd.DataFrame | None:
    """Выполняет запрос и возвращает результат как DataFrame (или None для DDL/DML)."""
    if engine == "duckdb":
        rel = con.execute(sql)
        if rel.description is None:
            return None
        return rel.fetchdf()
    cur = con.execute(sql)
    if cur.description is None:
        con.commit()
        return None
    return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])


def show(df: pd.DataFrame | None, max_rows: int = 30) -> None:
    if df is None:
        print("OK")
        return
    if df.empty:
        print("(пусто)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.max_rows", max_rows):
        print(df)
    print(f"[{len(df)} строк]")


# --------------------------------------------------------------------------- #
# Мета-команды и консоль
# --------------------------------------------------------------------------- #
def list_tables(con, engine: str) -> None:
    if engine == "duckdb":
        show(run_sql(con, engine, "SELECT table_name, estimated_size AS rows "
                                  "FROM duckdb_tables() ORDER BY table_name"))
    else:
        show(run_sql(con, engine, "SELECT name FROM sqlite_master "
                                  "WHERE type='table' ORDER BY name"))


def describe(con, engine: str, table: str) -> None:
    if engine == "duckdb":
        show(run_sql(con, engine, f"DESCRIBE {table}"))
    else:
        show(run_sql(con, engine, f"PRAGMA table_info({table})"))


def repl(con, engine: str) -> None:
    print(f"SQL-консоль ({engine}). Запрос завершается ';'. \\dt — таблицы, "
          f"\\d <table> — схема, \\q — выход.\n")
    buffer: list[str] = []
    while True:
        prompt = "sql> " if not buffer else "...> "
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return

        line = line.lstrip("﻿")          # BOM, если ввод пришёл из файла/пайпа
        stripped = line.strip()
        if not buffer and stripped.startswith("\\"):
            cmd, _, arg = stripped.partition(" ")
            if cmd in ("\\q", "\\quit"):
                return
            if cmd == "\\dt":
                list_tables(con, engine)
            elif cmd == "\\d":
                describe(con, engine, arg.strip())
            else:
                print(f"неизвестная команда: {cmd}")
            continue

        buffer.append(line)
        if not stripped.endswith(";"):
            continue

        sql = "\n".join(buffer).strip().rstrip(";")
        buffer.clear()
        if not sql:
            continue
        try:
            show(run_sql(con, engine, sql))
        except Exception as exc:                      # noqa: BLE001 — в песочнице
            print(f"ОШИБКА: {type(exc).__name__}: {exc}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SQL-песочница для подготовки к экзамену.")
    parser.add_argument("--engine", choices=["duckdb", "sqlite"], default="duckdb")
    parser.add_argument("--trades", type=Path, default=DATA_DIR / "task_1.parquet",
                        help="parquet со сделками; 'none' — не загружать")
    parser.add_argument("-c", "--command", help="выполнить один запрос и выйти")
    parser.add_argument("-f", "--file", type=Path, help="выполнить SQL-скрипт и выйти")
    args = parser.parse_args(argv)

    trades_path = None if str(args.trades).lower() == "none" else args.trades
    if trades_path is not None and not trades_path.is_file():
        print(f"Файл со сделками не найден: {trades_path}", file=sys.stderr)
        trades_path = None

    con = open_duckdb(trades_path) if args.engine == "duckdb" else open_sqlite(trades_path)

    if args.command:
        show(run_sql(con, args.engine, args.command))
        return 0

    if args.file:
        script = args.file.read_text(encoding="utf-8")
        for stmt in filter(str.strip, script.split(";")):
            print(f"\n--- {stmt.strip().splitlines()[0][:70]} ...")
            show(run_sql(con, args.engine, stmt))
        return 0

    repl(con, args.engine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
