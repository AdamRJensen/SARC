"""
SARC InfluxDB 3 ingestion script
---------------------------------
Downloads daily Campbell Scientific TOA5 CSV files from FTP,
writes raw data to InfluxDB, and computes 1-minute aggregates.

Tables written:
  - primary_fast       (primary fast table, 1s)
  - secondary_fast     (secondary fast table, 1s)
  - primary_slow       (primary slow table, 10s)
  - secondary_slow     (secondary slow table, 15s)
  - aggregated_1min    (1-min aggregates from all four tables)

Usage:
  python sarc_ingest.py --date 2026-03-25
  python sarc_ingest.py --date yesterday
  python sarc_ingest.py --date today
"""

import argparse
import ftplib
import io
import os
import re
from datetime import date, timedelta
import pandas as pd
import pvlib
import numpy as np
import warnings
from influxdb_client_3 import InfluxDBClient3

# ---------------------------------------------------------------------------
# Configuration — edit these or set as environment variables
# ---------------------------------------------------------------------------
FTP_HOST = os.environ["SARC_FTP_HOST"]
FTP_USER = os.environ["SARC_FTP_USER"]
FTP_PASSWORD = os.environ["SARC_FTP_PASSWORD"]
FTP_DIR = os.environ["SARC_FTP_DIR"]  # remote directory

INFLUXDB_HOST = os.environ["INFLUXDB3_HOST"]
INFLUXDB_TOKEN = os.environ["INFLUXDB3_AUTH_TOKEN"]
INFLUXDB_DB = os.environ["INFLUXDB3_DATABASE"]

# File name patterns (date formatted as YYYY-MM-DD)
FILE_PATTERNS = {
    "primary_fast": "{date}_00-00-00_primary_fast_table.csv",
    "secondary_fast": "{date}_00-00-00_secondary_fast_table.csv",
    "primary_slow": "{date}_00-00-00_primary_slow_table.csv",
    "secondary_slow": "{date}_00-00-00_secondary_slow_table.csv",
}

# Parameters that should NOT be averaged in the 1-min aggregation.
# Cumulative rain is taken as last value; per-interval rain is summed.
RAIN_CUMULATIVE = "Lufft_WS601_precipitation_cumulative_mm"
RAIN_INTERVAL = "Lufft_WS601_precipitation_mm"

EXCLUDE_COLUMNS = []  # placeholder for future columns to be excluded

COLUMN_RENAME_DICT = {
    # Commented out entries means that they were changed multiple times
    # "Time_stamp" was changed to "Time" on 2025-04-28
    # https://github.com/AdamRJensen/SARC/issues/25
    "Time_stamp": "Time",
    # Naming changes on 2025-03-26
    "Lufft_WS601_precip_intensive_mmh": "Lufft_WS601_precipitation_intensity_mmh",
    # 'Lufft_WS601_relative_pressure_hPa': 'Lufft_WS601_air_pressure_relative_hPa',
    # 'Lufft_WS601_abs_air_pressure_hPa': 'Lufft_WS601_air_pressure_absolute_hPa',
    "Lufft_WS601_relative_humidity_per": "Lufft_WS601_humidity_relative_per",
    # 'Lufft_WS601_precip_absolute_mm': 'Lufft_WS601_precipitation_absolute_mm',
    # 'Lufft_WS601_precip_different_mm': 'Lufft_WS601_precipitation_difference_mm',
    # Naming changes on 2025-04-28
    "MS80SH_S24053407_out_voltage_mV": "MS80SH_S24053407_GHI_mV",
    "DR30_65086_out_voltage_mV": "DR30_65086_DNI_mV",
    "SR30_23485_out_voltage_mV": "SR30_23485_GHI_mV",
    "SR300_45389_out_voltage_mV": "SR300_45389_GHI_mV",
    "SP522_1246_out_voltage_mV": "SP522_1246_GHI_mV",  # removed 2025-08-11
    "SP422_1843_out_voltage_mV": "SP422_1843_GHI_mV",
    "Lufft_WS601_precipitation_difference_mm": "Lufft_WS601_precipitation_mm",
    "Lufft_WS601_precipitation_absolute_mm": "Lufft_WS601_precipitation_cumulative_mm",
    "SP522_1246_GHI_Wm2": "SP522_1265_GHI_Wm2",  # removed 2025-08-11
    "SP522_1246_GHI_mV": "SP522_1265_GHI_mV",  # removed 2025-08-11
    "SP522_1246_heater_state": "SP522_1265_heater_state",  # removed 2025-08-11
    # Naming changes on 2025-05-02
    "Lufft_WS601_precip_type": "Lufft_WS601_precipitation_type",
    "Lufft_WS601_air_temperature_degC": "Lufft_WS601_temperature_air_degC",
    "Lufft_WS601_precip_absolute_mm": "Lufft_WS601_precipitation_cumulative_mm",
    "Lufft_WS601_precip_different_mm": "Lufft_WS601_precipitation_mm",
    # 'Lufft_WS601_precip_intensive_mmh': 'Lufft_WS601_precipitation_intensity_mmh',
    # 'Lufft_WS601_relative_humidity_per': 'Lufft_WS601_humidity_relative_per',
    "Lufft_WS601_relative_pressure_hPa": "Lufft_WS601_pressure_relative_air_hPa",
    "Lufft_WS601_air_pressure_relative_hPa": "Lufft_WS601_pressure_relative_air_hPa",
    "Lufft_WS601_abs_air_pressure_hPa": "Lufft_WS601_pressure_absolute_air_hPa",
    "Lufft_WS601_air_pressure_absolute_hPa": "Lufft_WS601_pressure_absolute_air_hPa",
    # Unknown date
    "SPN1_A270_Heater_ratio": "SPN1_A270_heater_ratio",
    "SPN1_A270_Sun_ratio": "SPN1_A270_sun_ratio",
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

longitude = 55.79064
longitude = 12.52505
altitude = 50


# ---------------------------------------------------------------------------
# APPLY CUSTOM NAN VALUES
# ---------------------------------------------------------------------------
def apply_pattern_nan_values(
    df: pd.DataFrame, pattern_nan_values: dict[str, list]
) -> pd.DataFrame:
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
        # quotechar='"',
        skiprows=(0, 2, 3),
    )

    df = df.rename(columns=COLUMN_RENAME_DICT)

    df = df.drop(columns=[c for c in EXCLUDE_COLUMNS if c in df.columns])

    # Convert integer columns to float
    df = df.apply(
        lambda col: col.astype("float64") if pd.api.types.is_integer_dtype(col) else col
    )

    df = apply_pattern_nan_values(df, PATTERN_NAN_VALUES)

    # Remove completely identical rows
    # this operation should come before setting index
    df = df[~df.duplicated(keep="first")]

    df["Time"] = pd.to_datetime(df["Time"], utc=True)
    df = df.set_index("Time")

    duplicated_indexes = df.index.duplicated().sum()
    if duplicated_indexes != 0:
        warnings.warn(
            f"File contained {duplicated_indexes} duplicated indexes!",
            UserWarning,
            stacklevel=2,
        )
        df = df[~df.index.duplicated(keep="first")]

    return df


# ---------------------------------------------------------------------------
# FTP download
# ---------------------------------------------------------------------------
def download_file(ftp: ftplib.FTP, filename: str) -> str | None:
    """Download a single file from FTP, return content as string or None."""
    buf = io.BytesIO()
    try:
        ftp.retrbinary(f"RETR {FTP_DIR}/{filename}", buf.write)
        return buf.getvalue().decode("utf-8", errors="replace")
    except ftplib.error_perm as e:
        print(f"  WARNING: Could not download {filename}: {e}")
        return None


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

    # Separate special columns if present
    special_cols = [
        c for c in [RAIN_CUMULATIVE, RAIN_INTERVAL] if c in combined.columns
    ]
    normal_cols = [c for c in combined.columns if c not in special_cols]

    agg_parts = []

    if normal_cols:
        agg_parts.append(combined[normal_cols].resample("1min").mean())

    if RAIN_CUMULATIVE in combined.columns:
        agg_parts.append(combined[[RAIN_CUMULATIVE]].resample("1min").last())

    if RAIN_INTERVAL in combined.columns:
        agg_parts.append(combined[[RAIN_INTERVAL]].resample("1min").sum())

    return pd.concat(agg_parts, axis=1, sort=False)


# ---------------------------------------------------------------------------
# CALCULATE SOLAR PARAMETERS
# ---------------------------------------------------------------------------
def get_solar_parameters(times, latitude, longitude, altitude):
    solar_parameters = pvlib.solarposition.get_solarposition(
        times, latitude, longitude, altitude
    )
    solar_parameters = solar_parameters.rename(columns=SOLAR_POSITION_COLUMN_DICT)
    solar_parameters = solar_parameters[SOLAR_POSITION_COLUMN_DICT.values()]
    solar_parameters["dni_extra"] = pvlib.irradiance.get_extra_radiation(times)
    cos_sza = np.cos(np.deg2rad(solar_parameters["solar_zenith"])).clip(lower=0)
    solar_parameters["ghi_extra"] = solar_parameters["dni_extra"] * cos_sza

    return solar_parameters.round(2)


# ---------------------------------------------------------------------------
# InfluxDB writing
# ---------------------------------------------------------------------------
def write_dataframe(client: InfluxDBClient3, df: pd.DataFrame, table: str) -> None:
    """Write a DataFrame to InfluxDB. Drops rows where all values are NaN."""
    df = df.dropna(how="all")
    if df.empty:
        print(f"Skipping {table} — no data after dropping all-NaN rows")
        return
    chunk_size = 4000
    total = 0
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        client.write(
            record=chunk,
            data_frame_measurement_name=table,
        )
        total += len(chunk)
        print(f"{table}: wrote {total:,}/{len(df):,} rows", end="\r")
    print(f"Wrote {total:,} rows to {table}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def resolve_date(date_str: str) -> date:
    if date_str == "today":
        return date.today()
    elif date_str == "yesterday":
        return date.today() - timedelta(days=1)
    else:
        return date.fromisoformat(date_str)


def main():
    parser = argparse.ArgumentParser(description="Ingest SARC data into InfluxDB")
    parser.add_argument(
        "--date",
        default="yesterday",
        help="Date to ingest: YYYY-MM-DD, 'today', or 'yesterday' (default: yesterday)",
    )
    args = parser.parse_args()

    target_date = resolve_date(args.date)
    date_str = target_date.strftime("%Y-%m-%d")
    print(f"Ingesting data for {date_str}")

    # Connect to FTP
    print(f"Connecting to FTP {FTP_HOST}...")
    ftp = ftplib.FTP(FTP_HOST)
    ftp.login(FTP_USER, FTP_PASSWORD)

    # Connect to InfluxDB
    client = InfluxDBClient3(
        host=INFLUXDB_HOST,
        token=INFLUXDB_TOKEN,
        database=INFLUXDB_DB,
        write_timeout=60_000,  # milliseconds
    )

    raw_frames = {}

    for table, pattern in FILE_PATTERNS.items():
        filename = pattern.format(date=date_str)
        print(f"Downloading {filename}...")
        content = download_file(ftp, filename)
        if content is None:
            continue

        df = parse_toa5(content)
        print(f"  Parsed {len(df):,} rows from {filename}")

        write_dataframe(client, df, table)
        raw_frames[table] = df

    ftp.quit()

    # Compute and write 1-minute aggregates
    if raw_frames:
        print("Computing 1-minute aggregates...")
        data_1min = aggregate_1min(raw_frames)
        solar_parameters = get_solar_parameters(
            data_1min.index, latitude=55.79064, longitude=12.52505, altitude=50
        )
        agg = pd.concat([data_1min, solar_parameters], axis=1, sort=False)
        write_dataframe(client, agg, "aggregated_1min")
    else:
        print("No data downloaded — skipping aggregation")

    client.close()
    print("Done.")


if __name__ == "__main__":
    main()
