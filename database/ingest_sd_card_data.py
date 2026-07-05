"""
SARC MySQL ingestion script — SD card backfill
-----------------------------------------------------
Reads historical Campbell Scientific TOA5 .dat files retrieved from the
datalogger SD cards, writes raw data to MySQL, then computes 1-minute
aggregates per calendar day across the ingested date range.

Each time the datalogger restarts it creates a new file series with a
timestamp prefix (e.g. 2025-04-08_12-54-18_primary_fast_table0.dat).
Large tables are split into numbered chunk files.

Tables written (database: sarc):
  - primary_fast       (primary fast table, 1s)
  - secondary_fast     (secondary fast table, 1s)
  - primary_slow       (primary slow table, 10s)
  - secondary_slow     (secondary slow table, 15s)
  - aggregated_1min    (1-min aggregates from all four tables)

Each raw table starts out with only a `time` PRIMARY KEY column. This
script automatically adds any measurement columns it encounters in the
source files (via ALTER TABLE ... ADD COLUMN ... ALGORITHM=INSTANT), so
new sensor channels showing up in the logger files never require a
manual schema change.

Usage:
  python sarc_ingest_sd_data.py

Requires:
  pip install mysql-connector-python pandas numpy pvlib --break-system-packages
"""

import argparse
import io
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import warnings

import mysql.connector
from mysql.connector import errorcode

# ---------------------------------------------------------------------------
# Configuration — edit these or set as environment variables
# ---------------------------------------------------------------------------
SD_DATA_DIR = os.environ.get(
    "SARC_SD_DATA_DIR",
    r"\\ait-pdfs.win.dtu.dk\Department\Cme\Section-ES\Solar future\SARC\Station_SD_data",
)

MYSQL_HOST     = os.environ.get("MYSQL_SARC_HOST", "localhost")
MYSQL_PORT     = int(os.environ.get("MYSQL_SARC_PORT", "3306"))
MYSQL_USER     = os.environ.get("MYSQL_SARC_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_SARC_PASSWORD", "LldfSdfD12k#k31ef8")
MYSQL_DATABASE = os.environ.get("MYSQL_SARC_DATABASE", "sarc")

# Raw table names (used both for writing and for querying back during aggregation)
RAW_TABLES = ["primary_fast", "secondary_fast", "primary_slow", "secondary_slow"]

# Glob patterns for each raw table type.
# Filenames follow: {session-timestamp}_{table}_table{N}.dat
TABLE_GLOBS = {
    "primary_fast":   "**/*_primary_fast_table*.dat",
    "primary_slow":   "**/*_primary_slow_table*.dat",
    "secondary_fast": "**/*_secondary_fast_table*.dat",
    "secondary_slow": "**/*_secondary_slow_table*.dat",
}

# Parameters that should NOT be averaged in the 1-min aggregation.
# Cumulative rain is taken as last value; per-interval rain is summed.
RAIN_CUMULATIVE = "Lufft_WS601_precipitation_cumulative_mm"
RAIN_INTERVAL   = "Lufft_WS601_precipitation_mm"

EXCLUDE_COLUMNS = [
    'MS80SHplus_1209_DHI_Wm2',
    'MS80SHplus_1209_DHI_uncorrected_Wm2',
    'MS80SHplus_1209_DNI_Wm2',
    'MS80SHplus_1209_DNI_uncorrected_Wm2',
    'MS80SHplus_1209_GHI_Wm2',
    'MS80SHplus_1209_GHI_reading_Wm2',
    'MS80SHplus_1209_band_offset_neg_Wm2',
    'MS80SHplus_1209_band_offset_pos_Wm2',
    'MS80SHplus_1209_rh_per',
    'MS80SHplus_1209_temperature_degC',
    'MS80SHplus_1209_tilt_x_deg',
    'MS80SHplus_1209_tilt_y_deg',
    'SPN1_A270_data(12)',
    'SPN1_A270_data(13)',
    'SPN1_A270_data(14)',
]

COLUMN_RENAME_DICT = {
    # Commented out entries means that they were changed multiple times
    # "Time_stamp" was changed to "Time" on 2025-04-28
    # https://github.com/AdamRJensen/SARC/issues/25
    'Time_stamp': 'Time',
    # Naming changes on 2025-03-26
    'Lufft_WS601_precip_intensive_mmh': 'Lufft_WS601_precipitation_intensity_mmh',
    # 'Lufft_WS601_relative_pressure_hPa': 'Lufft_WS601_air_pressure_relative_hPa',
    # 'Lufft_WS601_abs_air_pressure_hPa': 'Lufft_WS601_air_pressure_absolute_hPa',
    'Lufft_WS601_relative_humidity_per': 'Lufft_WS601_humidity_relative_per',
    # 'Lufft_WS601_precip_absolute_mm': 'Lufft_WS601_precipitation_absolute_mm',
    # 'Lufft_WS601_precip_different_mm': 'Lufft_WS601_precipitation_difference_mm',
    # Naming changes on 2025-04-28
    'MS80SH_S24053407_out_voltage_mV': 'MS80SH_S24053407_GHI_mV',
    'DR30_65086_out_voltage_mV': 'DR30_65086_DNI_mV',
    'SR30_23485_out_voltage_mV': 'SR30_23485_GHI_mV',
    'SR300_45389_out_voltage_mV': 'SR300_45389_GHI_mV',
    'SP522_1246_out_voltage_mV': 'SP522_1246_GHI_mV',  # removed 2025-08-11
    'SP422_1843_out_voltage_mV': 'SP422_1843_GHI_mV',
    'Lufft_WS601_precipitation_difference_mm': 'Lufft_WS601_precipitation_mm',
    'Lufft_WS601_precipitation_absolute_mm': 'Lufft_WS601_precipitation_cumulative_mm',
    'SP522_1246_GHI_Wm2': 'SP522_1265_GHI_Wm2',  # removed 2025-08-11
    'SP522_1246_GHI_mV': 'SP522_1265_GHI_mV',  # removed 2025-08-11
    'SP522_1246_heater_state': 'SP522_1265_heater_state',  # removed 2025-08-11
    # Naming changes on 2025-05-02
    'Lufft_WS601_precip_type': 'Lufft_WS601_precipitation_type',
    'Lufft_WS601_air_temperature_degC': 'Lufft_WS601_temperature_air_degC',
    'Lufft_WS601_precip_absolute_mm': 'Lufft_WS601_precipitation_cumulative_mm',
    'Lufft_WS601_precip_different_mm': 'Lufft_WS601_precipitation_mm',
    # 'Lufft_WS601_precip_intensive_mmh': 'Lufft_WS601_precipitation_intensity_mmh',
    # 'Lufft_WS601_relative_humidity_per': 'Lufft_WS601_humidity_relative_per',
    'Lufft_WS601_relative_pressure_hPa': 'Lufft_WS601_pressure_relative_air_hPa',
    'Lufft_WS601_air_pressure_relative_hPa': 'Lufft_WS601_pressure_relative_air_hPa',
    'Lufft_WS601_abs_air_pressure_hPa': 'Lufft_WS601_pressure_absolute_air_hPa',
    'Lufft_WS601_air_pressure_absolute_hPa': 'Lufft_WS601_pressure_absolute_air_hPa',
    # Unknown date
    'SPN1_A270_Heater_ratio': 'SPN1_A270_heater_ratio',
    'SPN1_A270_Sun_ratio': 'SPN1_A270_sun_ratio',
}

SOLAR_POSITION_COLUMN_DICT = {
    "apparent_zenith": "solar_zenith",
    "apparent_elevation": "solar_elevation",
    "azimuth": "solar_azimuth",
}

PATTERN_NAN_VALUES: dict[str, list] = {
    r".*": [-7999, -2147483648, -214748400.0, -21474840.0],
    "SPN1_A270_rh_per": [649.36],
    "SPN1_A270_temperature_degC": [608.51, 608.52],
    "SPN1_A270_heater_ratio": [-524288.0],
    "SPN1_A270_sun_ratio": [-2147484.0],
    "SR300_45389_GHI_mV": [10841.89],
}

# NOTE: original script had `longitude` assigned twice (the first was meant
# to be `latitude`). Fixed here, and aggregate_day() below now uses these
# constants instead of re-hardcoding the values.
latitude  = 55.79064
longitude = 12.52505
altitude  = 50


# ---------------------------------------------------------------------------
# APPLY CUSTOM NAN VALUES
# ---------------------------------------------------------------------------
def apply_pattern_nan_values(df: pd.DataFrame, pattern_nan_values: dict[str, list]) -> pd.DataFrame:
    """Replace values with NaN based on column name patterns."""
    for pattern, nan_vals in pattern_nan_values.items():
        matching_cols = [c for c in df.columns if re.fullmatch(pattern, c)]
        for col in matching_cols:
            df[col] = df[col].replace(nan_vals, np.nan)
    return df


# ---------------------------------------------------------------------------
# TOA5 parsing
# ---------------------------------------------------------------------------
def parse_toa5(content: str) -> pd.DataFrame:
    """
    Parse a Campbell Scientific CSV file in the TOA5 form.
    Skips rows (0, 2, 3) (station info, units, sampling type).
    Returns a DataFrame with a timezone-aware DatetimeIndex.
    """
    # Row index 1 contains column names (0-based)
    df = pd.read_csv(
        io.StringIO(content),
        na_values=["NAN", '"NAN"'],
        skiprows=(0, 2, 3),
    )

    df = df.rename(columns=COLUMN_RENAME_DICT)

    df = df.drop(columns=[c for c in EXCLUDE_COLUMNS if c in df.columns])

    # Convert integer columns to float
    df = df.apply(lambda col: col.astype('float64') if pd.api.types.is_integer_dtype(col) else col)

    df = apply_pattern_nan_values(df, PATTERN_NAN_VALUES)

    # Remove completely identical rows
    # this operation should come before setting index
    df = df[~df.duplicated(keep='first')]

    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df = df.set_index("Time")

    duplicated_indexes = df.index.duplicated().sum()
    if duplicated_indexes != 0:
        warnings.warn(f"File contained {duplicated_indexes} duplicated indexes!", UserWarning, stacklevel=2)
        df = df[~df.index.duplicated(keep='first')]

    return df


# ---------------------------------------------------------------------------
# SD card file discovery
# ---------------------------------------------------------------------------
def find_sd_files(source_dir: str) -> dict[str, list[Path]]:
    """
    Scan source_dir for SD card .dat files grouped by raw table name.
    Within each table, files are sorted by session timestamp then chunk index
    so data is ingested in chronological order.
    """
    root = Path(source_dir)
    result = {}
    for table, pattern in TABLE_GLOBS.items():
        files = list(root.glob(pattern))
        files.sort()
        result[table] = files
    return result


# ---------------------------------------------------------------------------
# 1-minute aggregation
# ---------------------------------------------------------------------------
def aggregate_1min(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all raw frames and compute 1-minute aggregates.
    - Most columns: mean
    - RAIN_CUMULATIVE: last value
    - RAIN_INTERVAL: sum
    """
    combined = pd.concat(frames.values(), axis=1, sort=False)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    # Separate special columns if present
    special_cols = [c for c in [RAIN_CUMULATIVE, RAIN_INTERVAL] if c in combined.columns]
    normal_cols  = [c for c in combined.columns if c not in special_cols]

    agg_parts = []

    if normal_cols:
        agg_parts.append(combined[normal_cols].resample("1min").mean())

    if RAIN_CUMULATIVE in combined.columns:
        agg_parts.append(
            combined[[RAIN_CUMULATIVE]].resample("1min").last()
        )

    if RAIN_INTERVAL in combined.columns:
        agg_parts.append(
            combined[[RAIN_INTERVAL]].resample("1min").sum()
        )

    return pd.concat(agg_parts, axis=1, sort=False)


# ---------------------------------------------------------------------------
# CALCULATE SOLAR PARAMETERS
# ---------------------------------------------------------------------------
def get_solar_parameters(times, latitude, longitude, altitude):
    solar_parameters = pvlib.solarposition.get_solarposition(
        times, latitude, longitude, altitude)
    solar_parameters = solar_parameters.rename(columns=SOLAR_POSITION_COLUMN_DICT)
    solar_parameters = solar_parameters[SOLAR_POSITION_COLUMN_DICT.values()]
    solar_parameters["dni_extra"] = pvlib.irradiance.get_extra_radiation(times)
    cos_sza = np.cos(np.deg2rad(solar_parameters["solar_zenith"])).clip(lower=0)
    solar_parameters["ghi_extra"] = solar_parameters["dni_extra"] * cos_sza

    return solar_parameters.round(2)


# ---------------------------------------------------------------------------
# MySQL connection + schema helpers
# ---------------------------------------------------------------------------
def get_connection():
    """Open a connection to the sarc MySQL database."""
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )


def _mysql_type_for_series(series: pd.Series) -> str:
    """Pick a MySQL column type for a pandas Series' dtype."""
    if pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(series):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(series):
        return "TINYINT(1)"
    # Fallback for text/categorical columns (e.g. precipitation type codes)
    return "VARCHAR(255)"


def ensure_columns(conn, table: str, df: pd.DataFrame) -> None:
    """
    Make sure every column in df exists on `table`. Any missing column is
    added with ALGORITHM=INSTANT, so this is safe to call before every
    write even on tables with years of existing data.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """,
        (MYSQL_DATABASE, table),
    )
    existing = {row[0] for row in cursor.fetchall()}

    missing = [c for c in df.columns if c not in existing]
    for col in missing:
        col_type = _mysql_type_for_series(df[col])
        print(f"  {table}: adding new column `{col}` ({col_type})")
        cursor.execute(
            f"ALTER TABLE `{table}` ADD COLUMN `{col}` {col_type}, ALGORITHM=INSTANT"
        )
    if missing:
        conn.commit()
    cursor.close()


def write_dataframe(conn, df: pd.DataFrame, table: str) -> None:
    """Write a DataFrame to a MySQL table, upserting on the `time` primary key."""
    df = df.dropna(how="all")
    if df.empty:
        print(f"Skipping {table} — no data after dropping all-NaN rows")
        return

    ensure_columns(conn, table, df)

    # Reset index into a `time` column, matching the table's primary key
    df = df.rename_axis("time").reset_index()
    # mysql-connector needs None, not NaN, for NULL values
    df = df.where(pd.notnull(df), None)

    columns = list(df.columns)
    col_list = ", ".join(f"`{c}`" for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    update_clause = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in columns if c != "time")

    insert_sql = (
        f"INSERT INTO `{table}` ({col_list}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )

    cursor = conn.cursor()
    chunk_size = 20000
    total = 0
    rows = list(df.itertuples(index=False, name=None))
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i:i + chunk_size]
        cursor.executemany(insert_sql, chunk)
        conn.commit()
        total += len(chunk)
        print(f"{table}: wrote {total:,}/{len(rows):,} rows", end="\r")
    cursor.close()
    print(f"Wrote {total:,} rows to {table}")


# ---------------------------------------------------------------------------
# Per-day aggregation using data queried back from MySQL
# ---------------------------------------------------------------------------
def read_raw_table(conn, table: str, start_str: str, end_str: str) -> pd.DataFrame:
    """Query one day of raw data from MySQL and return a DatetimeIndex DataFrame."""
    query = (
        f"SELECT * FROM `{table}` "
        f"WHERE `time` >= %s AND `time` < %s "
        f"ORDER BY `time`"
    )
    df = pd.read_sql(query, conn, params=(start_str, end_str))
    if df is None or df.empty:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time").rename_axis("Time").sort_index()


def aggregate_day(conn, day: date) -> None:
    """Read one calendar day of raw data from MySQL and write 1-minute aggregates."""
    start = datetime(day.year, day.month, day.day)
    end = start + timedelta(days=1)
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")

    raw_frames = {}
    for table in RAW_TABLES:
        df = read_raw_table(conn, table, start_str, end_str)
        df = df[~df.duplicated(keep="first")]
        if df.index.duplicated().any():
            print(f"Warning: {table} still contains duplicated indexes after dropping duplicates! ")
            df = df[~df.index.duplicated(keep="first")]
        if not df.empty:
            raw_frames[table] = df

    if not raw_frames:
        return

    data_1min = aggregate_1min(raw_frames)
    solar_parameters = get_solar_parameters(
        data_1min.index, latitude=latitude, longitude=longitude, altitude=altitude)
    agg = pd.concat([data_1min, solar_parameters], axis=1, sort=False)
    write_dataframe(conn, agg, "aggregated_1min")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Backfill SARC SD card data into MySQL"
    )
    parser.add_argument(
        "--source",
        default=SD_DATA_DIR,
        help=f"Directory containing Station_SD_data files (default: {SD_DATA_DIR})",
    )
    args = parser.parse_args()

    # Connect to MySQL
    try:
        conn = get_connection()
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("MySQL access denied — check SARC_MYSQL_USER / SARC_MYSQL_PASSWORD")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print(f"Database '{MYSQL_DATABASE}' does not exist")
        else:
            print(err)
        return

    print(f"Scanning {args.source} for SD card .dat files...")
    files_by_table = find_sd_files(args.source)

    overall_start = None
    overall_end   = None

    # --- Raw ingestion: process each file individually to keep memory bounded ---
    for table in RAW_TABLES:
        files = files_by_table.get(table, [])
        if not files:
            print(f"\n{table}: no files found, skipping")
            continue
        print(f"\n{table}: ingesting {len(files)} files")
        for path in files:
            print(f"  Reading {path.name}...")
            content = path.read_text(encoding="utf-8", errors="replace")
            df = parse_toa5(content)
            if df.empty:
                print(f"    (empty after parsing, skipping)")
                continue
            print(f"    Parsed {len(df):,} rows  "
                  f"({df.index.min()} → {df.index.max()})")
            # Write high resolution raw data (unaggregated) to MySQL
            write_dataframe(conn, df, table)

            if overall_start is None or df.index.min() < overall_start:
                overall_start = df.index.min()
            if overall_end is None or df.index.max() > overall_end:
                overall_end = df.index.max()

    if overall_start is None:
        print("\nNo data ingested — skipping aggregation")
        conn.close()
        return

    # --- Per-day aggregation: query raw data back and compute 1-min aggregates ---
    first_day = overall_start.date()
    last_day  = overall_end.date()
    print(f"\nComputing 1-minute aggregates from {first_day} to {last_day}...")
    day = first_day
    while day <= last_day:
        print(f"  Aggregating {day}...", end="\r")
        aggregate_day(conn, day)
        day += timedelta(days=1)
    print(f"\nAggregation complete ({(last_day - first_day).days + 1} days)")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
