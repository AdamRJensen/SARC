"""
Utility for querying InfluxDB 3 database with automatic time-based chunking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import pandas as pd


def get_influxdb3_data(
    client,
    table: str,
    columns: Sequence[str],
    start: str | datetime,
    end: str | datetime,
    freq: str = "1min",
    chunk_size: str = "1ME",
) -> pd.DataFrame:
    """Query InfluxDB 3 Enterprise over a date range, fetching one chunk at a time.

    Parameters
    ----------
    client:
        An ``influxdb3`` client instance (must support ``.query(..., mode="pandas")``).
    table:
        Measurement / table name to query.
    columns:
        Column names to fetch (excluding ``time``, which is always included).
    start:
        Inclusive start of the query range. ISO 8601 string or timezone-aware datetime.
    end:
        Exclusive end of the query range. ISO 8601 string or timezone-aware datetime.
    freq:
        Pandas offset alias used to regularise the index via ``asfreq``.
        Defaults to ``"1min"``.
    chunk_size:
        Pandas offset alias controlling the width of each sub-query.
        Defaults to ``"1ME"`` (one month, end-of-month anchored).
        Examples: ``"7D"``, ``"2W"``, ``"1ME"``, ``"QE"``.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC), one column per requested column, regularised to
        ``freq``. Gaps are filled with ``NaN``.

    Example
    -------
    >>> from influxdb3 import InfluxDBClient3
    >>> client = InfluxDBClient3(host="http://localhost:8181", database="mydb",
    ...                          token="MY_TOKEN")
    >>> df = get_influxdb3_data(
    ...     client,
    ...     table="aggregated_1min",
    ...     columns=["SMP22_200060_DHI_Wm2", "SR300_45389_GHI_Wm2"],
    ...     start="2025-04-01T00:00:00Z",
    ...     end="2026-06-01T00:00:00Z",  # exclusive
    ...     chunk_size="1ME",
    ... )
    """
    start_dt = _to_utc(start)
    end_dt = _to_utc(end)

    if start_dt >= end_dt:
        raise ValueError(f"start ({start_dt}) must be before end ({end_dt})")

    offset = pd.tseries.frequencies.to_offset(chunk_size)
    chunks: list[pd.DataFrame] = []
    chunk_start = start_dt

    while chunk_start < end_dt:
        # Advance by one offset step to find chunk_end
        chunk_end = min(
            (chunk_start + offset).to_pydatetime().replace(tzinfo=timezone.utc),
            end_dt,
        )

        quoted_cols = ", ".join(f'"{c}"' for c in columns)
        query = f"""
            SELECT time, {quoted_cols}
            FROM {table}
            WHERE time >= '{_fmt(chunk_start)}'
              AND time < '{_fmt(chunk_end)}'
        """

        chunk_df: pd.DataFrame = client.query(query, mode="pandas")

        if chunk_df is not None and not chunk_df.empty:
            chunk_df = chunk_df.set_index("time")
            chunk_df.index = pd.to_datetime(chunk_df.index, utc=True)
            chunks.append(chunk_df)

        chunk_start = chunk_end

    if not chunks:
        # Return an empty but correctly shaped DataFrame
        idx = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz="UTC")
        return pd.DataFrame(index=idx, columns=list(columns), dtype=float)

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df.asfreq(freq)
    return df


def write_influxdb3_data(
    client,
    df: pd.DataFrame,
    table: str,
    tag_columns: Sequence[str] | None = None,
    chunk_size: str = "1ME",
) -> None:
    """Write a DataFrame to InfluxDB 3 Enterprise, writing one chunk at a time.

    Parameters
    ----------
    client:
        An ``influxdb3`` client instance (must support ``.write(...)``).
    df:
        DataFrame with a UTC ``DatetimeIndex``. Each column is written as a
        field, except those listed in ``tag_columns``.
    table:
        Measurement / table name to write to.
    tag_columns:
        Column names in ``df`` to write as tags rather than fields.
        Defaults to ``None`` (no tag columns).
    chunk_size:
        Pandas offset alias controlling the width of each sub-write.
        Defaults to ``"1ME"`` (one month, end-of-month anchored).
        Examples: ``"7D"``, ``"2W"``, ``"1ME"``, ``"QE"``.

    Example
    -------
    >>> from influxdb3 import InfluxDBClient3
    >>> client = InfluxDBClient3(host="http://localhost:8181", database="mydb",
    ...                          token="MY_TOKEN")
    >>> write_influxdb3_data(
    ...     client,
    ...     df=df,
    ...     table="aggregated_1min",
    ...     chunk_size="1ME",
    ... )
    """
    if df.empty:
        return

    df = df.sort_index()
    start_dt = _to_utc(df.index.min())
    end_dt = _to_utc(df.index.max())

    offset = pd.tseries.frequencies.to_offset(chunk_size)
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = (chunk_start + offset).to_pydatetime().replace(tzinfo=timezone.utc)

        chunk_df = df.loc[(df.index >= chunk_start) & (df.index < chunk_end)]

        if not chunk_df.empty:
            client.write(
                record=chunk_df,
                data_frame_measurement_name=table,
                data_frame_tag_columns=list(tag_columns) if tag_columns else None,
            )

        chunk_start = chunk_end


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_utc(dt: str | datetime) -> datetime:
    """Parse an ISO 8601 string or datetime and ensure it is UTC-aware."""
    dt = pd.Timestamp(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt(dt: datetime) -> str:
    """Format a datetime as an ISO 8601 string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
