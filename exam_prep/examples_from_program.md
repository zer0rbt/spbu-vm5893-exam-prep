# Раздел 2 программы: разбор двух примеров практических заданий

## Задание 1. Распределение сумм забронированных билетов (SQL)

Схема (упрощённо, по образцу учебной БД авиаперевозок):

```sql
CREATE TABLE bookings (
    book_ref    CHAR(6) PRIMARY KEY,
    book_date   TIMESTAMPTZ NOT NULL,
    total_amount NUMERIC(10,2) NOT NULL
);
CREATE TABLE tickets (
    ticket_no   CHAR(13) PRIMARY KEY,
    book_ref    CHAR(6) NOT NULL REFERENCES bookings(book_ref),
    passenger_id VARCHAR(20) NOT NULL,
    passenger_name TEXT NOT NULL
);
CREATE TABLE ticket_flights (
    ticket_no   CHAR(13) NOT NULL REFERENCES tickets(ticket_no),
    flight_id   INT NOT NULL,
    fare_conditions VARCHAR(10) NOT NULL,
    amount      NUMERIC(10,2) NOT NULL,
    PRIMARY KEY (ticket_no, flight_id)
);
```

Задача: построить гистограмму — сколько бронирований попадает в каждый диапазон
стоимости шириной 100 000 рублей.

**Идея:** номер корзины = `floor(amount / 100000)`, границы восстанавливаются как
`bucket * 100000` и `(bucket + 1) * 100000`. Группируем по номеру корзины.

```sql
WITH b AS (
    SELECT total_amount,
           FLOOR(total_amount / 100000)::int AS bucket
    FROM   bookings
)
SELECT bucket * 100000                         AS range_from,
       (bucket + 1) * 100000                   AS range_to,
       COUNT(*)                                AS bookings_cnt,
       SUM(total_amount)                       AS bookings_sum,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS share_pct
FROM   b
GROUP  BY bucket
ORDER  BY bucket;
```

Замечания, которые стоит проговорить на экзамене:

* Пустые диапазоны в такой группировке пропускаются. Чтобы они присутствовали в
  выдаче с нулём, нужен левый JOIN со сгенерированным рядом корзин:

```sql
WITH bounds AS (
    SELECT FLOOR(MIN(total_amount) / 100000)::int AS lo,
           FLOOR(MAX(total_amount) / 100000)::int AS hi
    FROM bookings
),
grid AS (SELECT generate_series(lo, hi) AS bucket FROM bounds)
SELECT g.bucket * 100000       AS range_from,
       (g.bucket + 1) * 100000 AS range_to,
       COUNT(b.book_ref)       AS bookings_cnt,
       COALESCE(SUM(b.total_amount), 0) AS bookings_sum
FROM   grid g
LEFT   JOIN bookings b
       ON b.total_amount >= g.bucket * 100000
      AND b.total_amount <  (g.bucket + 1) * 100000
GROUP  BY g.bucket
ORDER  BY g.bucket;
```

* Если распределение нужно строить не по `bookings.total_amount`, а по суммам,
  собранным из перелётов, сумма агрегируется заранее:

```sql
WITH booking_amount AS (
    SELECT t.book_ref, SUM(tf.amount) AS total_amount
    FROM   ticket_flights tf
    JOIN   tickets t ON t.ticket_no = tf.ticket_no
    GROUP  BY t.book_ref
)
SELECT FLOOR(total_amount / 100000)::int * 100000 AS range_from,
       COUNT(*) AS bookings_cnt
FROM   booking_amount
GROUP  BY 1
ORDER  BY 1;
```

* Альтернатива в стандарте SQL — `width_bucket(total_amount, 0, 1000000, 10)` или
  `CASE WHEN ... THEN ... END`; идея та же.
* Граница включается слева и исключается справа (`[lo, hi)`) — это надо явно
  указать, чтобы не было двойного счёта.
* Для больших таблиц полезен индекс по `total_amount`, но при полной гистограмме
  всё равно будет seq scan — гистограмма читает все строки.

---

## Задание 2. Сумма частот пяти самых популярных слов (Python)

Задача: извлечь из файла (CSV) весь текст, привести слова к единой нормальной
форме и вычислить сумму классических term frequency пяти самых частых слов.

Классический TF слова $w$: $\mathrm{tf}(w) = \dfrac{n_w}{\sum_k n_k}$, где $n_w$ —
число вхождений слова, знаменатель — общее число слов в тексте. Значит ответ —
это доля пяти самых частых слов в тексте, число из отрезка $[0, 1]$.

```python
"""Сумма term frequency пяти самых частых слов в CSV-файле. Python 3.10+."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # только буквенные последовательности


def iter_text(path: Path, encoding: str = "utf-8") -> Iterator[str]:
    """Выдаёт текст всех ячеек CSV, включая заголовок, по одной строке файла."""
    with path.open(encoding=encoding, newline="") as fh:
        for row in csv.reader(fh):
            for cell in row:
                if cell:
                    yield cell


def normalize(word: str, lemmatizer=None) -> str:
    """Приводит слово к единой нормальной форме.

    Базовый вариант — приведение к нижнему регистру. Если доступна pymorphy2,
    слово лемматизируется (кот/коту/котом -> кот), что и требуется в задании.
    """
    word = word.lower()
    if lemmatizer is None:
        return word
    return lemmatizer.parse(word)[0].normal_form


def make_lemmatizer():
    """Возвращает морфологический анализатор или None, если библиотеки нет."""
    try:
        import pymorphy2
    except ImportError:
        return None
    return pymorphy2.MorphAnalyzer()


def top_terms(path: Path, top_n: int = 5, encoding: str = "utf-8") -> tuple[list[tuple[str, float]], float]:
    """Возвращает top_n слов с их TF и сумму этих TF.

    На пустом файле возвращает ([], 0.0) — деления на ноль не происходит.
    """
    lemmatizer = make_lemmatizer()
    cache: dict[str, str] = {}          # лемматизация дорогая, кешируем
    counter: Counter[str] = Counter()

    for chunk in iter_text(path, encoding):
        for raw in WORD_RE.findall(chunk):
            lemma = cache.get(raw)
            if lemma is None:
                lemma = normalize(raw, lemmatizer)
                cache[raw] = lemma
            counter[lemma] += 1

    total = sum(counter.values())
    if total == 0:
        return [], 0.0

    top = [(word, cnt / total) for word, cnt in counter.most_common(top_n)]
    return top, sum(tf for _, tf in top)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сумма TF пяти самых частых слов в CSV.")
    parser.add_argument("path", type=Path)
    parser.add_argument("-n", "--top", type=int, default=5)
    parser.add_argument("--encoding", default="utf-8")
    args = parser.parse_args(argv)

    if not args.path.is_file():
        print(f"Файл не найден: {args.path}", file=sys.stderr)
        return 1

    top, total_tf = top_terms(args.path, args.top, args.encoding)
    if not top:
        print("Текстовых данных в файле нет; сумма частот равна 0.")
        return 0

    for rank, (word, tf) in enumerate(top, start=1):
        print(f"{rank}. {word:<20} tf = {tf:.6f}")
    print(f"\nСумма частот {len(top)} самых популярных слов: {total_tf:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Что важно упомянуть в ответе (за это и ставят баллы):

* «Нормальная форма» — это лемматизация, а не просто `lower()`. Для русского
  языка — pymorphy2/pymorphy3 или Natasha, для английского — стемминг Портера или
  лемматизатор NLTK/spaCy. Код должен работать и без сторонней библиотеки, поэтому
  предусмотрен запасной вариант.
* Токенизация регулярным выражением `[^\W\d_]+` отбрасывает числа и пунктуацию;
  можно дополнительно убирать стоп-слова (`и`, `в`, `не`) — тогда результат
  осмысленнее, но условие задачи это не требует, и об изменении постановки надо
  сказать явно.
* Файл читается потоково, целиком в память не загружается — это существенно для
  больших данных.
* Вырожденные случаи: пустой файл, файл без текста, слов меньше пяти — программа
  не должна падать (в критериях оценивания это отдельный пункт на 15 баллов).
* Кеш лемматизации важен по производительности: морфологический разбор на порядок
  дороже, чем поиск в словаре.

---

## Нужен ли pandas

Зависит от задачи. Правило: **табличная агрегация — pandas; извлечение текста и
потоковый разбор — стандартная библиотека.**

### Задача про торги — да, pandas обязателен

`trades_analysis.py` без него писать бессмысленно: сами данные лежат в parquet
(его читает `pandas.read_parquet`, под капотом pyarrow), а всё задание —
группировки, ресемплинг по временным интервалам, скользящие вычисления. На
чистом Python это сотни строк и минуты работы вместо секунд.

### Задача про term frequency — нет, и вариант без pandas лучше

Тут pandas не нужен: `csv` + `collections.Counter` читают файл потоково и не
зависят от структуры таблицы. Вариант на pandas короче, но хрупче:

```python
import pandas as pd

def top_terms_pandas(path: str, top_n: int = 5) -> pd.Series:
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    words = (
        df.stack()
          .str.lower()
          .str.findall(r"[^\W\d_]+")
          .explode()
          .dropna()
    )
    if words.empty:
        return pd.Series(dtype=float)
    return words.value_counts(normalize=True).head(top_n)
```

На корректном CSV он даёт ровно тот же ответ, что и версия из стандартной
библиотеки (проверено: 0.818182 на одном и том же файле). Но:

* CSV с «рваными» строками (в разных строках разное число полей — например,
  запятая внутри незакавыченного текста) роняет `read_csv` с `ParserError`,
  тогда как `csv.reader` спокойно отдаёт три ячейки вместо двух;
* пустой файл — `EmptyDataError`, это исключение нужно ловить отдельно;
* весь файл загружается в память, а `.explode()` создаёт строку на каждое слово —
  на большом тексте это кратный расход памяти;
* `header=None` обязателен, иначе первая строка станет именами столбцов и слова
  из заголовка не попадут в подсчёт.

То есть за счёт краткости теряется устойчивость к вырожденным входным данным —
а это отдельный пункт критериев на 15 баллов. Если берёте pandas, обработку
`EmptyDataError` и `ParserError` (`on_bad_lines="skip"`) нужно писать явно.

### Организационное

На экзамене нет интернета, а `pip install` во время испытания сделать не выйдет.
Поэтому окружение надо подготовить заранее: `pandas`, `pyarrow` (для parquet),
`numpy`, и, если планируете лемматизацию, `pymorphy2`/`pymorphy3` со словарями.
Проверьте перед экзаменом, что `import pandas, pyarrow, pymorphy2` отрабатывает
офлайн. Если библиотеки не окажется — решение на чистой стандартной библиотеке
должно быть запасным вариантом, поэтому обе версии здесь и приведены.

---

## Где прогонять SQL

Писать SQL вслепую не нужно — есть `sql_playground.py`, он поднимает базу в
памяти и даёт консоль:

```
python sql_playground.py                  # интерактивная консоль (duckdb)
python sql_playground.py -c "SELECT 1"    # один запрос
python sql_playground.py -f queries.sql   # скрипт
python sql_playground.py --engine sqlite  # другой диалект
```

Таблицы внутри:

* `trades` — сделки из `data/task_1.parquet` (можно подменить через `--trades`);
* `bookings`, `tickets`, `ticket_flights` — синтетическая БД авиаперевозок по
  образцу из Задания 1, 5000 бронирований с логнормальными суммами. Данные
  генерируются с фиксированным seed, так что результаты воспроизводимы.

В консоли: `;` завершает запрос, `\dt` — список таблиц, `\d <table>` — схема,
`\q` — выход.

### Два движка и зачем это нужно

**DuckDB** (`pip install duckdb`) — почти полный SQL:2016: оконные функции, CTE
(в т.ч. рекурсивные), `generate_series`, `QUALIFY`, `FILTER`, читает parquet
напрямую. Синтаксис близок к PostgreSQL, так что на нём удобно отлаживать
«экзаменационный» SQL.

**SQLite** — встроен в стандартную библиотеку Python, ставить нечего. Диалект
беднее: нет типа `TIMESTAMP` (даты хранятся строками), нет функции
`generate_series` (нужен рекурсивный CTE), `FULL OUTER JOIN` появился только в
3.39, `::int` не работает — нужен `CAST(... AS INT)`. Полезен как проверка: если
запрос работает в обоих движках, он почти наверняка переносим.

### Проверенные различия диалектов

Запрос с сеткой корзин из Задания 1 в чистом виде — это PostgreSQL. В DuckDB
`generate_series` возвращает список, и его нужно развернуть:

```sql
-- DuckDB
grid AS (SELECT UNNEST(generate_series(lo, hi)) AS bucket FROM bounds)
```

В SQLite сетка строится рекурсивным CTE:

```sql
WITH RECURSIVE bounds AS (
    SELECT CAST(MIN(total_amount) / 100000 AS INT) AS lo,
           CAST(MAX(total_amount) / 100000 AS INT) AS hi
    FROM bookings
),
grid(bucket, hi) AS (
    SELECT lo, hi FROM bounds
    UNION ALL
    SELECT bucket + 1, hi FROM grid WHERE bucket < hi
)
SELECT g.bucket * 100000 AS range_from, COUNT(b.book_ref) AS cnt
FROM   grid g
LEFT   JOIN bookings b
       ON b.total_amount >= g.bucket * 100000
      AND b.total_amount <  (g.bucket + 1) * 100000
GROUP  BY g.bucket
ORDER  BY g.bucket;
```

На сгенерированных данных обе версии дают одинаковый результат, и в нём хорошо
видно, ради чего нужен `LEFT JOIN` с сеткой: корзины 1 400 000 и 1 500 000
пустые, и простой `GROUP BY` их просто пропускает, а версия с сеткой показывает
нули. Это ровно тот случай, о котором стоит написать в комментарии к решению.
