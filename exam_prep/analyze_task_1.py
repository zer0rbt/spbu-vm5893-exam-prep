"""Разведочный анализ task_1.parquet средствами pandas.

Скрипт печатает отчёт в stdout и, если указан --md, сохраняет его же в
markdown-файл. Используется только pandas/numpy — так же, как это придётся
делать на практической части экзамена.

Запуск:
    python analyze_task_1.py data/task_1.parquet --md analysis_task_1.md
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


class Report:
    """Накапливает markdown-текст отчёта и одновременно печатает его."""

    def __init__(self) -> None:
        self._buf = io.StringIO()

    def head(self, level: int, text: str) -> None:
        self.line(f"\n{'#' * level} {text}\n")

    def line(self, text: str = "") -> None:
        print(text)
        self._buf.write(text + "\n")

    def table(self, df: pd.DataFrame, floatfmt: str = ".6g") -> None:
        self.line(df.to_markdown(floatfmt=floatfmt))
        self.line()

    def text(self) -> str:
        return self._buf.getvalue()


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["turnover"] = df["price"] * df["quantity"]
    return df


# --------------------------------------------------------------------------- #
def section_overview(rep: Report, df: pd.DataFrame) -> None:
    rep.head(2, "1. Обзор датасета")
    start, end = df["timestamp"].min(), df["timestamp"].max()
    rep.line(f"* строк (сделок): **{len(df):,}**".replace(",", " "))
    rep.line(f"* столбцов: {list(df.columns)}")
    rep.line(f"* период: **{start} — {end}** ({end - start})")
    rep.line(f"* товаров N: **{df['product_id'].nunique()}** "
             f"(id от {df['product_id'].min()} до {df['product_id'].max()})")
    rep.line(f"* суммарный оборот: **{df['turnover'].sum():,.2f}**".replace(",", " "))
    rep.line(f"* объём памяти в RAM: {df.memory_usage(deep=True).sum() / 1024**2:.1f} МБ")
    rep.line()

    rep.line("Типы и пропуски:")
    dtypes = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "nunique": df.nunique(),
    })
    rep.table(dtypes)


def section_quality(rep: Report, df: pd.DataFrame) -> None:
    rep.head(2, "2. Качество данных")

    dupes = int(df.duplicated().sum())
    non_positive = int(((df["price"] <= 0) | (df["quantity"] <= 0)).sum())
    sorted_ok = bool(df["timestamp"].is_monotonic_increasing)
    same_ts = int(df["timestamp"].duplicated().sum())

    rep.line(f"* полные дубликаты строк: **{dupes:,}**".replace(",", " "))
    rep.line(f"* неположительные price/quantity: **{non_positive}**")
    rep.line(f"* данные отсортированы по timestamp: **{sorted_ok}**")
    rep.line(f"* сделок с неуникальной меткой времени: **{same_ts:,}** "
             f"({same_ts / len(df):.1%})".replace(",", " "))
    rep.line()
    rep.line("Дубликаты здесь ожидаемы: одна заявка исполняется несколькими сделками "
             "с одной меткой времени, поэтому удалять их нельзя — это разные сделки.")
    rep.line()

    rep.line("Шаг цены (тик) по товарам — минимальная ненулевая разность соседних цен:")
    ticks = {}
    for pid, grp in df.groupby("product_id"):
        diffs = grp["price"].sort_values().diff().dropna()
        diffs = diffs[diffs > 0]
        ticks[pid] = {
            "tick": diffs.min() if not diffs.empty else np.nan,
            "уровней цены": grp["price"].nunique(),
        }
    rep.table(pd.DataFrame(ticks).T.rename_axis("product_id"))

    gaps = df["timestamp"].diff().dropna()
    rep.line("Промежутки между соседними сделками (по всему датасету):")
    rep.line(f"* медиана: {gaps.median()}")
    rep.line(f"* среднее: {gaps.mean()}")
    rep.line(f"* максимум: {gaps.max()}")
    rep.line()


def section_products(rep: Report, df: pd.DataFrame) -> pd.DataFrame:
    rep.head(2, "3. Разрез по товарам")

    g = df.groupby("product_id")
    stats = g.agg(
        trades=("price", "size"),
        price_min=("price", "min"),
        price_mean=("price", "mean"),
        price_max=("price", "max"),
        price_std=("price", "std"),
        first_price=("price", "first"),
        last_price=("price", "last"),
        volume=("quantity", "sum"),
        turnover=("turnover", "sum"),
    )
    stats["vwap"] = stats["turnover"] / stats["volume"]
    stats["cv_%"] = stats["price_std"] / stats["price_mean"] * 100
    stats["range_%"] = (stats["price_max"] / stats["price_min"] - 1) * 100
    stats["change_%"] = (stats["last_price"] / stats["first_price"] - 1) * 100
    stats["trades_%"] = stats["trades"] / stats["trades"].sum() * 100
    stats["turnover_%"] = stats["turnover"] / stats["turnover"].sum() * 100

    rep.line("Цены и доли:")
    rep.table(stats[["trades", "trades_%", "price_min", "vwap", "price_max",
                     "cv_%", "range_%", "change_%", "turnover", "turnover_%"]])

    rep.line("Размер сделки (quantity) — распределение по товарам:")
    qstats = g["quantity"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    rep.table(qstats)

    skew = g["quantity"].apply(lambda s: s.skew())
    rep.line("Коэффициент асимметрии quantity: "
             + ", ".join(f"товар {pid} — {v:.1f}" for pid, v in skew.items()))
    rep.line()
    rep.line("Распределение размера сделки сильно скошено вправо: медиана в разы "
             "меньше среднего, значит поток состоит из множества мелких сделок и "
             "редких крупных. Для «типичного» размера сделки надо брать медиану, "
             "а не среднее.")
    rep.line()

    top_share = g["turnover"].apply(
        lambda s: s.nlargest(max(1, int(len(s) * 0.01))).sum() / s.sum() * 100
    )
    rep.line("Доля оборота, которую дают 1% крупнейших сделок: "
             + ", ".join(f"товар {pid} — {v:.1f}%" for pid, v in top_share.items()))
    rep.line()
    return stats


def section_time(rep: Report, df: pd.DataFrame) -> pd.DataFrame:
    rep.head(2, "4. Динамика во времени")

    idx = df.set_index("timestamp")
    candles = (
        idx.groupby("product_id")
        .resample("1h")
        .agg(open=("price", "first"), high=("price", "max"), low=("price", "min"),
             close=("price", "last"), volume=("quantity", "sum"),
             turnover=("turnover", "sum"), trades=("price", "size"))
    )
    candles["close"] = candles.groupby(level="product_id")["close"].ffill()

    empty = candles.groupby(level="product_id")["trades"].apply(lambda s: (s == 0).sum())
    total = candles.groupby(level="product_id").size()
    rep.line(f"Часовых интервалов на товар: {int(total.iloc[0])}. "
             "Интервалов без единой сделки: "
             + ", ".join(f"товар {pid} — {int(v)}" for pid, v in empty.items()))
    rep.line()

    closes = candles["close"].unstack("product_id").sort_index().ffill()
    rets = np.log(closes / closes.shift(1)).dropna(how="all")

    dyn = pd.DataFrame({
        "close_first": closes.iloc[0],
        "close_last": closes.iloc[-1],
        "change_%": (closes.iloc[-1] / closes.iloc[0] - 1) * 100,
        "hourly_vol_%": rets.std() * 100,
        "ann_vol_%": rets.std() * np.sqrt(24 * 365) * 100,
        "max_drawdown_%": (closes / closes.cummax() - 1).min() * 100,
        "max_run_up_%": (closes / closes.cummin() - 1).max() * 100,
    })
    rep.line("Итоги периода по ценам закрытия часовых свечей:")
    rep.table(dyn)

    rep.line("Дневная динамика цены закрытия (последняя сделка дня):")
    daily = closes.resample("1D").last()
    daily_pct = (daily / daily.iloc[0] - 1) * 100
    rep.table(daily_pct.iloc[::5], floatfmt=".2f")
    rep.line("(показана каждая пятая дата, значения — накопленное изменение к 1 апреля, %)")
    rep.line()

    ac = rets.apply(lambda s: s.autocorr(lag=1))
    rep.line("Автокорреляция часовых доходностей (лаг 1): "
             + ", ".join(f"товар {pid} — {v:+.3f}" for pid, v in ac.items()))
    rep.line("Значения около нуля означают, что направление следующего часа по "
             "предыдущему практически не предсказывается — обычная картина для цен.")
    rep.line()
    return rets


def section_seasonality(rep: Report, df: pd.DataFrame) -> None:
    rep.head(2, "5. Сезонность активности")

    by_hour = df.groupby(df["timestamp"].dt.hour).agg(
        trades=("price", "size"), turnover=("turnover", "sum"))
    by_hour["trades_%"] = by_hour["trades"] / by_hour["trades"].sum() * 100
    by_hour.index.name = "час UTC"
    rep.line("По часам суток:")
    rep.table(by_hour, floatfmt=".2f")

    peak, low = by_hour["trades"].idxmax(), by_hour["trades"].idxmin()
    ratio = by_hour["trades"].max() / by_hour["trades"].min()
    rep.line(f"Пик активности — {peak}:00, минимум — {low}:00, "
             f"отношение {ratio:.2f}. Разброс умеренный: биржа работает круглосуточно, "
             "выраженного «торгового дня» нет.")
    rep.line()

    names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    by_dow = df.groupby(df["timestamp"].dt.dayofweek).agg(
        trades=("price", "size"), turnover=("turnover", "sum"))
    by_dow.index = [names[i] for i in by_dow.index]
    by_dow["trades_%"] = by_dow["trades"] / by_dow["trades"].sum() * 100
    rep.line("По дням недели:")
    rep.table(by_dow, floatfmt=".2f")

    by_day = df.resample("1D", on="timestamp").agg(
        trades=("price", "size"), turnover=("turnover", "sum"))
    busiest = by_day["trades"].idxmax()
    rep.line(f"Самый активный день — {busiest.date()}: "
             f"{int(by_day['trades'].max()):,} сделок при медиане "
             f"{int(by_day['trades'].median()):,} — "
             f"в {by_day['trades'].max() / by_day['trades'].median():.1f} раза выше обычного."
             .replace(",", " "))
    rep.line()


def section_corr(rep: Report, rets: pd.DataFrame) -> None:
    rep.head(2, "6. Связь между товарами")

    corr = rets.corr()
    rep.line("Корреляция часовых логарифмических доходностей (Пирсон):")
    rep.table(corr, floatfmt=".3f")

    rep.line("Ранговая корреляция Спирмена (устойчива к выбросам):")
    rep.table(rets.corr(method="spearman"), floatfmt=".3f")

    pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .sort_values(ascending=False)
    )
    x_id, y_id = pairs.index[0]
    rep.line(f"Максимальная связь — товары {x_id} и {y_id}: r = {pairs.iloc[0]:.3f}.")
    rep.line()

    pair = rets[[x_id, y_id]].dropna()
    x, y = pair[x_id], pair[y_id]
    x_c, y_c = x - x.mean(), y - y.mean()
    slope = float((x_c * y_c).sum() / (x_c**2).sum())
    intercept = float(y.mean() - slope * x.mean())
    r = float(x.corr(y))
    resid = y - (intercept + slope * x)
    se_slope = float(np.sqrt((resid**2).sum() / (len(pair) - 2) / (x_c**2).sum()))
    t_stat = slope / se_slope

    rep.line("Парная линейная регрессия МНК (проверка гипотезы о линейной связи):")
    rep.line()
    rep.line("```")
    rep.line(f"r_{y_id} = {intercept:+.6f} {slope:+.4f} * r_{x_id}")
    rep.line(f"R^2      = {r**2:.4f}")
    rep.line(f"se(b1)   = {se_slope:.4f},  t = {t_stat:.2f},  n = {len(pair)}")
    rep.line("```")
    rep.line()
    rep.line(f"Коэффициент значим (|t| = {abs(t_stat):.1f} много больше критических 1.96), "
             f"но R^2 = {r**2:.2f} — связь есть, а объясняющая сила модели низкая: "
             "движения товаров в основном независимы.")
    rep.line()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Анализ task_1.parquet на pandas.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--md", type=Path, default=None, help="куда сохранить отчёт")
    args = parser.parse_args(argv)

    df = load(args.path)

    rep = Report()
    rep.line(f"# Анализ датасета `{args.path.name}`")
    rep.line()
    rep.line("Отчёт сгенерирован скриптом `analyze_task_1.py` (pandas + numpy).")

    section_overview(rep, df)
    section_quality(rep, df)
    section_products(rep, df)
    rets = section_time(rep, df)
    section_seasonality(rep, df)
    section_corr(rep, rets)

    if args.md:
        args.md.write_text(rep.text(), encoding="utf-8")
        print(f"\nОтчёт сохранён: {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

