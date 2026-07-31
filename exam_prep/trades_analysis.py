"""Анализ истории торгов на электронной бирже (task_1.parquet / task_2.parquet).

Язык программирования: Python 3.10+.

Входные данные — DataFrame со столбцами:
    timestamp  : datetime64[ns] — время сделки
    price      : float          — цена единицы товара
    quantity   : float          — количество единиц товара
    product_id : int            — код товара, 0..N-1

Что считается:
    1. Валидация и общее описание датасета.
    2. Описательные статистики по каждому товару (в т.ч. VWAP и оборот).
    3. OHLCV-свечи: разбиение периода на равные интервалы.
    4. Динамика цены: доходности, волатильность, максимальная просадка.
    5. Корреляция доходностей товаров и парная линейная регрессия (МНК).
    6. Профиль активности по часам суток.

Запуск:
    python trades_analysis.py task_1.parquet --freq 1h --out report
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "price", "quantity", "product_id")


# --------------------------------------------------------------------------- #
# Загрузка и валидация
# --------------------------------------------------------------------------- #
def load_trades(path: str | Path) -> pd.DataFrame:
    """Читает parquet со сделками и приводит его к каноническому виду.

    Отбрасывает строки с пропусками и неположительными ценой/количеством,
    сортирует по времени. Возвращает пустой DataFrame с нужной схемой,
    если во входных данных не осталось ни одной валидной сделки.
    """
    df = pd.read_parquet(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"В файле нет обязательных столбцов: {sorted(missing)}")

    df = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["product_id"] = df["product_id"].astype("int64")

    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]

    # Оборот сделки — понадобится почти во всех агрегациях.
    df["turnover"] = df["price"] * df["quantity"]

    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class DatasetInfo:
    """Сводка по датасету в целом."""

    n_trades: int
    n_products: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    total_turnover: float

    def describe(self) -> str:
        if self.n_trades == 0:
            return "Датасет пуст: валидных сделок нет."
        span = self.end - self.start
        return (
            f"Сделок: {self.n_trades:,}\n"
            f"Товаров (N): {self.n_products}\n"
            f"Период: {self.start} — {self.end} (длительность {span})\n"
            f"Суммарный оборот: {self.total_turnover:,.2f}"
        ).replace(",", " ")


def dataset_info(df: pd.DataFrame) -> DatasetInfo:
    if df.empty:
        return DatasetInfo(0, 0, None, None, 0.0)
    return DatasetInfo(
        n_trades=len(df),
        n_products=int(df["product_id"].nunique()),
        start=df["timestamp"].iloc[0],
        end=df["timestamp"].iloc[-1],
        total_turnover=float(df["turnover"].sum()),
    )


# --------------------------------------------------------------------------- #
# 1. Описательные статистики по товарам
# --------------------------------------------------------------------------- #
def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Статистики по каждому товару.

    VWAP (средневзвешенная по объёму цена) = sum(p_i * q_i) / sum(q_i) —
    более честная «средняя цена товара», чем простое среднее по сделкам,
    так как учитывает размер сделки.
    """
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("product_id", sort=True)
    summary = grouped.agg(
        trades=("price", "size"),
        price_mean=("price", "mean"),
        price_std=("price", "std"),
        price_min=("price", "min"),
        price_median=("price", "median"),
        price_max=("price", "max"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        volume=("quantity", "sum"),
        turnover=("turnover", "sum"),
    )
    summary["vwap"] = summary["turnover"] / summary["volume"]
    # Коэффициент вариации — безразмерная мера разброса, позволяет
    # сравнивать товары с разным масштабом цены.
    summary["cv"] = summary["price_std"] / summary["price_mean"]
    summary["change_pct"] = (summary["last_price"] / summary["first_price"] - 1) * 100
    summary["turnover_share_pct"] = summary["turnover"] / summary["turnover"].sum() * 100
    return summary


# --------------------------------------------------------------------------- #
# 2. Разбиение периода на равные интервалы: OHLCV
# --------------------------------------------------------------------------- #
def ohlcv(df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Свечи (open/high/low/close/volume) по каждому товару за интервал `freq`.

    Реализация ресемплинга: индекс — время, группировка по товару и
    временному бину фиксированной длины.
    """
    if df.empty:
        return pd.DataFrame()

    indexed = df.set_index("timestamp")
    candles = (
        indexed.groupby("product_id")
        .resample(freq)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("quantity", "sum"),
            turnover=("turnover", "sum"),
            trades=("price", "size"),
        )
    )
    candles["vwap"] = candles["turnover"] / candles["volume"]
    # Интервалы без сделок: объём 0, цену переносим с предыдущей свечи.
    candles["close"] = candles.groupby(level="product_id")["close"].ffill()
    return candles


def interval_statistics(candles: pd.DataFrame) -> pd.DataFrame:
    """Описательные статистики по интервальным метрикам для каждого товара."""
    if candles.empty:
        return pd.DataFrame()
    return candles.groupby(level="product_id")[["close", "volume", "trades"]].describe()


# --------------------------------------------------------------------------- #
# 3. Динамика цены: доходности, волатильность, просадка
# --------------------------------------------------------------------------- #
def close_matrix(candles: pd.DataFrame) -> pd.DataFrame:
    """Широкая таблица «время × товар» с ценами закрытия."""
    if candles.empty:
        return pd.DataFrame()
    return candles["close"].unstack(level="product_id").sort_index().ffill()


def returns_matrix(closes: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    """Доходности по интервалам.

    Логарифмические доходности r_t = ln(P_t / P_{t-1}) удобны тем, что
    аддитивны по времени и приближённо симметричны.
    """
    if closes.empty:
        return pd.DataFrame()
    rets = np.log(closes / closes.shift(1)) if log else closes.pct_change()
    return rets.dropna(how="all")


def price_dynamics(closes: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    """Изменение цены, волатильность и максимальная просадка по товарам."""
    if closes.empty or rets.empty:
        return pd.DataFrame()

    first, last = closes.iloc[0], closes.iloc[-1]
    # Максимальная просадка: наибольшее относительное падение от достигнутого максимума.
    drawdown = (closes / closes.cummax() - 1).min() * 100

    return pd.DataFrame(
        {
            "first_close": first,
            "last_close": last,
            "change_pct": (last / first - 1) * 100,
            "volatility_per_interval_pct": rets.std() * 100,
            "mean_return_pct": rets.mean() * 100,
            "max_drawdown_pct": drawdown,
        }
    )


# --------------------------------------------------------------------------- #
# 4. Корреляционный анализ и парная линейная регрессия (МНК)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegressionResult:
    """Результат оценки модели y = b0 + b1 * x методом наименьших квадратов."""

    x_name: str
    y_name: str
    slope: float
    intercept: float
    r: float
    r_squared: float
    n: int

    def __str__(self) -> str:
        return (
            f"{self.y_name} = {self.intercept:+.6f} {self.slope:+.4f} * {self.x_name}   "
            f"(r = {self.r:+.3f}, R^2 = {self.r_squared:.3f}, n = {self.n})"
        )


def fit_pair_regression(x: pd.Series, y: pd.Series) -> RegressionResult | None:
    """МНК-оценки b1 = cov(x, y) / var(x), b0 = mean(y) - b1 * mean(x).

    Возвращает None для вырожденного случая (меньше двух точек или
    нулевая дисперсия x), чтобы не делить на ноль.
    """
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(pair) < 2:
        return None

    x_c = pair["x"] - pair["x"].mean()
    y_c = pair["y"] - pair["y"].mean()
    sxx = float((x_c**2).sum())
    syy = float((y_c**2).sum())
    if sxx == 0 or syy == 0:
        return None

    sxy = float((x_c * y_c).sum())
    slope = sxy / sxx
    intercept = float(pair["y"].mean() - slope * pair["x"].mean())
    r = sxy / np.sqrt(sxx * syy)
    return RegressionResult(
        x_name=str(x.name),
        y_name=str(y.name),
        slope=slope,
        intercept=intercept,
        r=float(r),
        r_squared=float(r**2),
        n=len(pair),
    )


def strongest_pair(rets: pd.DataFrame) -> tuple[object, object] | None:
    """Пара товаров с максимальной по модулю корреляцией доходностей."""
    if rets.shape[1] < 2:
        return None
    corr = rets.corr().abs()
    # copy=True обязателен: в pandas 3.0 .to_numpy() может вернуть read-only view.
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, np.nan)
    if np.isnan(values).all():
        return None
    i, j = np.unravel_index(np.nanargmax(values), values.shape)
    return corr.index[i], corr.columns[j]


# --------------------------------------------------------------------------- #
# 5. Профиль активности
# --------------------------------------------------------------------------- #
def hourly_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Распределение числа сделок и оборота по часам суток."""
    if df.empty:
        return pd.DataFrame()
    by_hour = df.groupby(df["timestamp"].dt.hour).agg(
        trades=("price", "size"),
        turnover=("turnover", "sum"),
    )
    by_hour.index.name = "hour"
    by_hour["trades_share_pct"] = by_hour["trades"] / by_hour["trades"].sum() * 100
    return by_hour


# --------------------------------------------------------------------------- #
# Отчёт
# --------------------------------------------------------------------------- #
def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def build_report(path: str | Path, freq: str, out_dir: Path | None) -> None:
    df = load_trades(path)
    info = dataset_info(df)

    _section(f"Файл: {Path(path).name}")
    print(info.describe())
    if df.empty:
        print("Дальнейший анализ невозможен.")
        return

    _section("1. Описательные статистики по товарам")
    summary = product_summary(df)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.round(6))

    _section(f"2. Интервальные свечи OHLCV (шаг {freq})")
    candles = ohlcv(df, freq=freq)
    print(f"Всего интервалов: {len(candles)}")
    print(candles.head(5).round(6))

    _section(f"3. Описательные статистики по интервалам (шаг {freq})")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(interval_statistics(candles).round(4))

    _section("4. Динамика цены за период")
    closes = close_matrix(candles)
    rets = returns_matrix(closes)
    print(price_dynamics(closes, rets).round(4))

    _section("5. Корреляция логарифмических доходностей товаров")
    print(rets.corr().round(3))
    pair = strongest_pair(rets)
    if pair is not None:
        x_id, y_id = pair
        model = fit_pair_regression(rets[x_id].rename(f"r_{x_id}"), rets[y_id].rename(f"r_{y_id}"))
        print("\nПарная линейная регрессия для наиболее связанной пары (МНК):")
        print(f"  {model}" if model else "  вырожденный случай, оценка невозможна")

    _section("6. Активность по часам суток")
    print(hourly_activity(df).round(2))

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(path).stem
        summary.to_csv(out_dir / f"{stem}_products.csv")
        candles.to_csv(out_dir / f"{stem}_ohlcv_{freq}.csv")
        rets.corr().to_csv(out_dir / f"{stem}_returns_corr.csv")
        print(f"\nCSV-файлы отчёта сохранены в {out_dir.resolve()}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ истории торгов из parquet-файла.")
    parser.add_argument("paths", nargs="+", help="пути к parquet-файлам")
    parser.add_argument("--freq", default="1h", help="длина интервала, напр. 15min, 1h, 1D")
    parser.add_argument("--out", default=None, help="каталог для выгрузки CSV-отчётов")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out) if args.out else None
    for path in args.paths:
        build_report(path, freq=args.freq, out_dir=out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
