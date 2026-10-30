"""
SARC MySQL ingestion script — SD card backfill + FTP daily files
----------------------------------------------------------------
Reads Campbell Scientific TOA5 data files, writes raw data to MySQL, then
computes 1-minute aggregates per calendar day across the ingested date range.

Every measurement column is stored as DOUBLE. Aggregation methods are
declared in AGGREGATIONS below. That dict holds only the exceptions;
every other channel is averaged.

Three source modes are supported (--mode is required; there is no default):

  sd: historical .dat files retrieved from the datalogger SD
    cards. Each time the datalogger restarts it creates a new file series
    with a timestamp prefix (e.g. 2025-04-08_12-54-18_primary_fast_table0.dat).
    Large tables are split into numbered chunk files. --start-date/--end-date
    restrict which rows are ingested; a session file whose timestamp falls
    after --end-date is skipped without being read.

  ftp: daily .csv files published to the SARC FTP server, one file per
    table per calendar day (e.g. 2025-06-01_00-00-00_primary_fast_table.csv).
    Useful both for routine daily syncing and for backfilling holes in a
    specific table over a specific date range.

  reaggregate: no ingestion at all — recompute 1-minute aggregates from raw
    data already in MySQL. Use after changing aggregation logic, or to rebuild
    aggregated_1min from scratch with --rebuild (drops and recreates the
    table; raw tables are never touched).

  logbook: no measurement ingestion — read the station logbook spreadsheet,
    derive 1-minute quality flags (cleaning, soiling per component) and write
    them into aggregated_1min alongside the measurements. Independent of the
    raw tables, so it can be run whenever the logbook is updated. Note that
    --mode reaggregate --rebuild drops these columns along with the rest of
    the table, so re-run this mode afterwards.

Tables written (database: sarc):
  - primary_fast       (primary fast table, 1s)
  - secondary_fast     (secondary fast table, 1s)
  - primary_slow       (primary slow table, 10s)
  - secondary_slow     (secondary slow table, 15s)
  - aggregated_1min    (1-min aggregates from all four tables, plus the
                        logbook flag columns)

Each raw table starts out with only a `time` PRIMARY KEY column. This script
automatically adds any measurement columns it encounters (via ALTER TABLE ...
ADD COLUMN ... ALGORITHM=INSTANT), so new sensor channels showing up in the
logger files never require a manual schema change.

Usage:
  python sarc_ingest.py --mode sd
  python sarc_ingest.py --mode sd --source <dir>
  python sarc_ingest.py --mode sd --start-date 2025-06-01 --end-date 2025-06-30
  python sarc_ingest.py --mode ftp --start-date 2025-06-01 --end-date 2025-06-10
  python sarc_ingest.py --mode ftp --start-date 2025-06-01 --tables primary_slow
  python sarc_ingest.py --mode reaggregate                # whole raw record
  python sarc_ingest.py --mode reaggregate --start-date 2025-06-01 --end-date 2025-06-02
  python sarc_ingest.py --mode reaggregate --rebuild      # drop + recreate + refill
  python sarc_ingest.py --mode logbook                    # refresh logbook flags
  python sarc_ingest.py --mode logbook --start-date 2025-06-01
  python sarc_ingest.py --mode logbook --dry-run          # derive, write nothing
  python sarc_ingest.py --mode schema                     # report unexpected types

Requires:
  pip install pymysql pandas numpy pvlib openpyxl
"""

import argparse
import ftplib
import io
import os
import re
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import pymysql
from pymysql.constants import ER

# ---------------------------------------------------------------------------
# Aggregation declarations
# ---------------------------------------------------------------------------
# Every measurement column is stored as DOUBLE and averaged over each minute.
# This dict lists only the channels that need something other than a mean.
# Supported methods: mean, median, min, max, first, last, std, var, sum,
# circular_mean.

DEFAULT_AGG = "mean"

AGGREGATIONS: dict[str, str] = {
    "Lufft_WS601_precipitation_cumulative_mm": "last",
    "Lufft_WS601_precipitation_mm":            "sum",
    "Lufft_WS601_precipitation_type":          "max",
    "Lufft_WS601_wind_direction_deg":          "circular_mean",
    "SP422_1843_heater_state":                 "max",
    "MS80SH_S24053407_heater_alert":           "max",
    "SP522_1265_heater_state":                 "max",
    "SolarBandC3_SN50013_sun":                 "max",
}


MASK_COLUMN_PATTERNS = [
    # Primary fast
    "SHP1_185163_",
    "SMP10_248585_",
    "SMP10_196704_",
    "SMP12_233555_",
    "SMP22_200057_",
    "SMP22_200060_",
    "SGR3_200107_",
    "CHP1_140049_",
    "CGR4_170223_",
    "CMP11_128767_GHI_mV",
    "CMP11_128758_DHI_mV",
    # Secondary fast
    "DR30_65086_DNI",
    "DR30_65086_tilt_deg",
    "MS80SH_S24053407_",
    "SR30_23485_GHI",
    "SR300_45389_GHI_",
    "SR300_45389_temperature_degC",
    "SRD100_14401_DHI_Wm2",
    "SRD100_14401_temperature_degC",
    "SRD100_14401_sensor_",
    "SP422_1843_GHI_",
    "SP522_1265_GHI_",
    "SP510_3882_GHI_mV",
    "Licor_PY116375_GHI_mV",
    "StarSchenk_7773_GHI_mV",
    "SunVue30_112_",
]


def aggregation(name: str) -> str:
    """Aggregation method for `name`; mean unless declared otherwise."""
    return AGGREGATIONS.get(name, DEFAULT_AGG)


# Default type for measurement columns.
COLUMN_TYPE = "DOUBLE"

# Columns that are stored as something other than COLUMN_TYPE, keyed by column
# name. Populated by the logbook section further down with the flag columns;
# set FLAG_TYPE to "DOUBLE" there to make the whole table uniform again.
COLUMN_TYPES: dict[str, str] = {}


def column_type(name: str) -> str:
    """MySQL type used when a column is added on demand."""
    return COLUMN_TYPES.get(name, COLUMN_TYPE)


# Resampler methods that can be called directly by name.
SIMPLE_AGGS = frozenset({"mean", "median", "min", "max", "first", "last", "std", "var"})

AGG_TABLE = "aggregated_1min"


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
MYSQL_PASSWORD = os.environ.get("MYSQL_SARC_PASSWORD")  # no default: set in env
MYSQL_DATABASE = os.environ.get("MYSQL_SARC_DATABASE", "sarc")

# FTP connection for daily files (only required when running --mode ftp;
# looked up lazily in get_ftp_connection() so sd-mode runs don't need these set)
FTP_HOST     = os.environ.get("SARC_FTP_HOST", "")
FTP_PORT     = int(os.environ.get("SARC_FTP_PORT", "21"))
FTP_USER     = os.environ.get("SARC_FTP_USER", "")
FTP_PASSWORD = os.environ.get("SARC_FTP_PASSWORD", "")
FTP_DIR      = os.environ.get("SARC_FTP_DIR", "/")  # remote directory

# Raw table names (used both for writing and for querying back during aggregation)
RAW_TABLES = ["primary_fast", "secondary_fast", "primary_slow", "secondary_slow"]

# Glob patterns for each raw table type (sd mode).
# Filenames follow: {session-timestamp}_{table}_table{N}.dat
SD_TABLE_GLOBS = {
    "primary_fast":   "primary/*_primary_fast_table*.dat",
    "primary_slow":   "primary/*_primary_slow_table*.dat",
    "secondary_fast": "secondary/*_secondary_fast_table*.dat",
    "secondary_slow": "secondary/*_secondary_slow_table*.dat",
}

# Daily filename patterns on the FTP server (ftp mode). `{date}` is filled
# in as YYYY-MM-DD.
FILE_PATTERNS = {
    "primary_fast":   "{date}_00-00-00_primary_fast_table.csv",
    "secondary_fast": "{date}_00-00-00_secondary_fast_table.csv",
    "primary_slow":   "{date}_00-00-00_primary_slow_table.csv",
    "secondary_slow": "{date}_00-00-00_secondary_slow_table.csv",
}

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
    'MS80plus_S24067011_rh_per ',  # not logged due to additional modbus call
    'PH1_190116_DNI_raw_Wm2',
    'PR1_190116_DHI_raw_Wm2',
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
    # Rename ventilation speed columns (removes sensor type from name)
    "CVF4_193087_speed_GHI_Hz": "CVF4_193087_speed_Hz",
    "CVF4_193086_speed_DHI_Hz": "CVF4_193086_speed_Hz",
    "CVF4_247678_speed_LWD_Hz": "CVF4_247678_speed_Hz",
    # SRD100 sensor are not thermopiles (renamed in datalogger on 2026-10-28)
    "SRD100_14401_thermopile_1x1_Wm2": "SRD100_14401_sensor_1x1_Wm2",
    "SRD100_14401_thermopile_1x2_Wm2": "SRD100_14401_sensor_1x2_Wm2",
    "SRD100_14401_thermopile_1x3_Wm2": "SRD100_14401_sensor_1x3_Wm2",
    "SRD100_14401_thermopile_2x1_Wm2": "SRD100_14401_sensor_2x1_Wm2",
    "SRD100_14401_thermopile_2x2_Wm2": "SRD100_14401_sensor_2x2_Wm2",
    "SRD100_14401_thermopile_2x3_Wm2": "SRD100_14401_sensor_2x3_Wm2",
    "SRD100_14401_thermopile_3x1_Wm2": "SRD100_14401_sensor_3x1_Wm2",
    "SRD100_14401_thermopile_3x2_Wm2": "SRD100_14401_sensor_3x2_Wm2",
    "SRD100_14401_thermopile_3x3_Wm2": "SRD100_14401_sensor_3x3_Wm2",
}

SOLAR_POSITION_COLUMN_MAP = {
    "apparent_zenith": "solar_zenith",
    "apparent_elevation": "solar_elevation",
    "azimuth": "solar_azimuth",
}

PATTERN_NAN_VALUES: dict[str, list] = {
    r".*": [-9999, -999.9, -7999, -2147483648, -214748400.0, -21474840.0],
    "SR30_": [65535],  # max value of unsigned 16-bit int
    "SPN1_A270_rh_per": [649.36],
    "SPN1_A270_temperature_degC": [608.51, 608.52],
    "SPN1_A270_heater_ratio": [-524288.0],
    "SPN1_A270_sun_ratio": [-2147484.0],
    "SR300_45389_GHI_mV": [10841.89],
    # Single corrupted sample in secondary_slow; the channel is 0 everywhere else.
    "SP422_1843_heater_state": [698.3754],
    "Lufft_WS601_wind_": [3276.7],  # wind/compass parameters, maximum signed 16-bit integer
}

LATITUDE  = 55.79064
LONGITUDE = 12.52505
ALTITUDE  = 50


# ---------------------------------------------------------------------------
# Logbook configuration
# ---------------------------------------------------------------------------
# Google Sheets export endpoint for the station logbook.
LOGBOOK_URL = ("https://docs.google.com/spreadsheets/d/"
               "11HGFfdIwgw48tLejh3nTuS204q5MC6n18jJW-pSLFfU"
               "/export?format=xlsx&gid=1885804395")

# Logbook times are written in station local time; everything in the database
# is UTC.
LOGBOOK_TIMEZONE = "Europe/Copenhagen"

# Entries before this date describe the construction period and are ignored.
# Naive because read_logbook() returns naive UTC, like the rest of the database.
LOGBOOK_RECORD_START = pd.Timestamp("2025-04-01")

# Event type (in the "Event type" column) that counts as a cleaning.
CLEANING_EVENT_TYPE = "Cleaning / regular inspection"

# A soiling observation flags backwards to the previous cleaning, or this many
# days if no cleaning was logged in between.
CLEANING_DAYS_THRESHOLD = 3

# Soiling levels considered severe enough to flag the preceding period.
SOILING_LEVELS_TO_FLAG = ("3", "4")

# Components with a "Soiling level (before cleaning) [X]" column.
SOILING_COMPONENTS = ("ghi", "dhi", "dni")

# Flag columns written into aggregated_1min. TINYINT(1) is one byte and is
# what MySQL itself means by BOOLEAN; the driver returns 0/1 and NULL for
# minutes the logbook says nothing about. Set this to "DOUBLE" if you would
# rather every column in the table have the same type.
FLAG_TYPE = "TINYINT(1)"

FLAG_COLUMNS = ["cleaning_flag"] + [f"soiling_flag_{c}" for c in SOILING_COMPONENTS]

# Registered here so ensure_columns() and check_schema() know these are not
# DOUBLE. Everything else in the database still uses COLUMN_TYPE.
COLUMN_TYPES.update({column: FLAG_TYPE for column in FLAG_COLUMNS})


# ---------------------------------------------------------------------------
# APPLY CUSTOM NAN VALUES
# ---------------------------------------------------------------------------
def apply_pattern_nan_values(df: pd.DataFrame,
                             pattern_nan_values: dict[str, list]) -> pd.DataFrame:
    """Replace values with NaN based on column name patterns."""
    for pattern, nan_vals in pattern_nan_values.items():
        matching_cols = [c for c in df.columns if re.fullmatch(pattern, c)]
        for col in matching_cols:
            df[col] = df[col].replace(nan_vals, np.nan)
    return df


def coerce_numeric(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """
    Force every measurement column to float64.

    Values that cannot be parsed become NaN, and are reported rather than
    silently flipping the column to object dtype (which is what previously
    caused numeric channels to be created as VARCHAR in MySQL).
    """
    for col in df.columns:
        original = df[col]
        coerced = pd.to_numeric(original, errors="coerce")
        lost = coerced.isna() & original.notna()
        if lost.any():
            examples = pd.unique(original[lost])[:3]
            warnings.warn(
                f"{source or 'input'}: column {col!r} had {int(lost.sum())} "
                f"non-numeric values set to NaN, e.g. {list(examples)}",
                UserWarning,
                stacklevel=2,
            )
        df[col] = coerced.astype("float64")
    return df


# ---------------------------------------------------------------------------
# TOA5 parsing
# ---------------------------------------------------------------------------
def parse_toa5(content: str, source: str = "") -> pd.DataFrame:
    """
    Parse a Campbell Scientific CSV file in the TOA5 form.
    Skips rows (0, 2, 3) (station info, units, sampling type).
    Returns a DataFrame with a DatetimeIndex and float64 columns.
    """
    # Row index 1 contains column names (0-based)
    df = pd.read_csv(
        io.StringIO(content),
        na_values=["NAN", '"NAN"'],
        skiprows=(0, 2, 3),
    )

    df = df.rename(columns=COLUMN_RENAME_DICT)
    df = df.drop(columns=[c for c in EXCLUDE_COLUMNS if c in df.columns])
    df = apply_pattern_nan_values(df, PATTERN_NAN_VALUES)

    # Remove completely identical rows
    # this operation should come before setting index
    df = df[~df.duplicated(keep='first')]

    df["Time"] = pd.to_datetime(df["Time"])
    df = df.set_index("Time")

    duplicated_indexes = df.index.duplicated().sum()
    if duplicated_indexes != 0:
        warnings.warn(f"File contained {duplicated_indexes} duplicated indexes!",
                      UserWarning, stacklevel=2)
        df = df[~df.index.duplicated(keep='first')]

    return coerce_numeric(df, source=source)


# ---------------------------------------------------------------------------
# SD card file discovery
# ---------------------------------------------------------------------------
SESSION_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}_")


def session_date(path: Path) -> date | None:
    """
    Date from an SD filename's session prefix (2025-04-08_12-54-18_...), or
    None if the name doesn't carry one.
    """
    match = SESSION_TIMESTAMP_RE.match(path.name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def find_sd_files(source_dir: str, end_date: date | None = None
                  ) -> dict[str, list[Path]]:
    """
    Scan source_dir for SD card .dat files grouped by raw table name.
    Within each table, files are sorted by session timestamp then chunk index
    so data is ingested in chronological order.

    Files whose session started after end_date are skipped without being read.
    Files that started earlier are kept even when end_date is set, because a
    session can run for weeks and the filename says nothing about where it
    ends — those are trimmed row-wise after parsing instead.
    """
    root = Path(source_dir)
    result = {}
    for table, pattern in SD_TABLE_GLOBS.items():
        files = sorted(root.glob(pattern))
        if end_date is not None:
            kept = []
            for path in files:
                started = session_date(path)
                if started is not None and started > end_date:
                    continue
                kept.append(path)
            if len(kept) != len(files):
                print(f"{table}: skipping {len(files) - len(kept)} file(s) "
                      f"starting after {end_date}")
            files = kept
        result[table] = files
    return result


# ---------------------------------------------------------------------------
# 1-minute aggregation
# ---------------------------------------------------------------------------
def circular_mean(values: pd.Series) -> float:
    """Vector mean of angles in degrees, returned on [0, 360)."""
    radians = np.deg2rad(pd.to_numeric(values, errors="coerce").dropna().to_numpy())
    if radians.size == 0:
        return np.nan
    degrees = np.rad2deg(np.arctan2(np.sin(radians).mean(),
                                    np.cos(radians).mean())) % 360.0
    # Floating point can push a result that is really 0 up to exactly 360.0
    return float(0.0 if degrees >= 360.0 - 1e-9 else degrees)


def mask_gaps(df, max_gap="10s", rule="1min", patterns=None):
    """
    Set sub-minute data points to NAN if the minute they belong to has long gaps.
    A "long gap" is defined by max_gap.
    """
    max_gap = pd.Timedelta(max_gap)

    if patterns is not None:
        combined = re.compile("|".join(f"(?:{p})" for p in patterns))
        checked_cols = df.columns[df.columns.str.contains(combined)]
    else:
        checked_cols = df.columns

    df = df.copy()
    rule_td = pd.tseries.frequencies.to_offset(rule)

    index_floor = df.index.floor(rule)
    for col in checked_cols:
        real = df[col].dropna()
        if real.empty:
            continue
        bins = real.index.floor(rule)

        # Internal gaps
        bins_s = pd.Series(bins, index=real.index)
        diffs = real.index.to_series().diff()
        cross_bin = bins_s.ne(bins_s.shift(1, fill_value=bins_s.iloc[0] - rule_td))
        diffs[cross_bin] = pd.NaT
        
        internal_gap = diffs.groupby(bins).max()

        # Head gap: distance from bin start to first sample in bin
        head_gap = (real.index - bins).to_series(index=real.index).groupby(bins).min()

        # Tail gap: distance from last sample in bin to bin end
        tail_gap = (bins + rule_td - real.index).to_series(index=real.index).groupby(bins).min()

        largest_gap = pd.concat([internal_gap, head_gap, tail_gap], axis=1).max(axis=1)
        bad_bins = largest_gap[largest_gap > max_gap].index

        mask = index_floor.isin(bad_bins)
        df.loc[mask, col] = float("nan")

    return df


def aggregate_1min(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine all raw frames and resample to 1 minute, using each channel's
    declared aggregation method (mean unless listed in AGGREGATIONS).
    """
    combined = pd.concat(frames.values(), axis=1, sort=False)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    combined = mask_gaps(combined, max_gap="10s", rule="1min",
                         patterns=MASK_COLUMN_PATTERNS)

    by_method: dict[str, list[str]] = {}
    for col in combined.columns:
        by_method.setdefault(aggregation(col), []).append(col)

    parts = []
    for method, cols in by_method.items():
        block = combined[cols].resample("1min")
        if method == "circular_mean":
            parts.append(block.agg(circular_mean))
        elif method == "sum":
            # min_count=1 so a minute with no observations stays NaN rather
            # than becoming a spurious 0 (matters for interval rainfall).
            parts.append(block.sum(min_count=1))
        elif method in SIMPLE_AGGS:
            parts.append(getattr(block, method)())
        else:
            raise ValueError(
                f"Unknown aggregation {method!r} declared for columns {cols}"
            )

    aggregated = pd.concat(parts, axis=1, sort=False)
    return aggregated[[c for c in combined.columns if c in aggregated.columns]]


# ---------------------------------------------------------------------------
# CALCULATE SOLAR PARAMETERS
# ---------------------------------------------------------------------------
def get_solar_parameters(times, latitude, longitude, altitude):
    solar_parameters = pvlib.solarposition.get_solarposition(
        times, latitude, longitude, altitude)
    solar_parameters = solar_parameters.rename(columns=SOLAR_POSITION_COLUMN_MAP)
    solar_parameters = solar_parameters[list(SOLAR_POSITION_COLUMN_MAP.values())]
    solar_parameters["dni_extra"] = pvlib.irradiance.get_extra_radiation(times)
    cos_sza = np.cos(np.deg2rad(solar_parameters["solar_zenith"])).clip(lower=0)
    solar_parameters["ghi_extra"] = solar_parameters["dni_extra"] * cos_sza
    return solar_parameters.round(2)


# ---------------------------------------------------------------------------
# MySQL connection + schema helpers
# ---------------------------------------------------------------------------
def get_connection():
    """Open a connection to the sarc MySQL database."""
    if not MYSQL_PASSWORD:
        raise SystemExit(
            "MYSQL_SARC_PASSWORD is not set. Export it in the environment "
            "rather than hardcoding credentials in this file."
        )
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
    )


# Cache of {table: {column: mysql_data_type}}, invalidated when columns are added.
_COLUMN_TYPES: dict[str, dict[str, str]] = {}


def get_column_types(conn, table: str) -> dict[str, str]:
    """Column name -> INFORMATION_SCHEMA.DATA_TYPE for one table (cached)."""
    if table not in _COLUMN_TYPES:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """,
            (MYSQL_DATABASE, table),
        )
        _COLUMN_TYPES[table] = {name: dtype for name, dtype in cursor.fetchall()}
        cursor.close()
    return _COLUMN_TYPES[table]


def ensure_columns(conn, table: str, df: pd.DataFrame) -> None:
    """
    Make sure every column in df exists on `table`. Missing columns are added
    with their declared type (DOUBLE for measurements, FLAG_TYPE for logbook
    flags) using ALGORITHM=INSTANT, so this is safe to call before every write
    even on tables with years of existing data.
    """
    existing = get_column_types(conn, table)
    missing = [c for c in df.columns if c not in existing]
    if not missing:
        return

    cursor = conn.cursor()
    for col in missing:
        col_type = column_type(col)
        print(f"  {table}: adding new column `{col}` ({col_type})")
        cursor.execute(
            f"ALTER TABLE `{table}` ADD COLUMN `{col}` {col_type}, "
            f"ALGORITHM=INSTANT"
        )
    conn.commit()
    cursor.close()
    _COLUMN_TYPES.pop(table, None)  # force a refresh on next read


def check_schema(conn) -> None:
    """Report any column not stored as its declared type."""
    total = 0
    for table in RAW_TABLES + [AGG_TABLE]:
        try:
            columns = get_column_types(conn, table)
        except pymysql.err.MySQLError:
            continue
        wrong = {}
        for col, dtype in columns.items():
            if col == "time":
                continue
            # INFORMATION_SCHEMA.DATA_TYPE carries no length, so TINYINT(1)
            # has to be compared as `tinyint`.
            expected = column_type(col).split("(")[0].lower()
            if dtype.lower() != expected:
                wrong[col] = (dtype, column_type(col))
        if wrong:
            total += len(wrong)
            print(f"\n{table}: {len(wrong)} column(s) with an unexpected type")
            for col, (dtype, expected) in sorted(wrong.items()):
                print(f"    {col}: {dtype.upper()} (expected {expected})")

    if total:
        print(f"\n{total} column(s) to convert. {AGG_TABLE} can be fixed with "
              f"--mode reaggregate --rebuild followed by --mode logbook; raw "
              f"tables need ALTER TABLE ... MODIFY COLUMN ... {COLUMN_TYPE}.")
    else:
        print("All columns match their declared types.")


def write_dataframe(conn, df: pd.DataFrame, table: str) -> None:
    """Write a DataFrame to a MySQL table, upserting on the `time` primary key."""
    df = df.dropna(how="all")
    if df.empty:
        print(f"Skipping {table} — no data after dropping all-NaN rows")
        return

    ensure_columns(conn, table, df)

    # Reset index into a `time` column, matching the table's primary key
    df = df.rename_axis("time").reset_index()
    # pymysql rejects NaN outright (mysql-connector silently tolerated it), so
    # it must become None. astype(object) first is required: on a float64
    # column, pandas silently coerces None back into NaN in .where() since a
    # numpy float64 array has nowhere else to put a NULL.
    df = df.astype(object).where(df.notna(), None)

    columns = list(df.columns)
    col_list = ", ".join(f"`{c}`" for c in columns)
    placeholders = ", ".join(["%s"] * len(columns))
    update_clause = ", ".join(f"`{c}`=VALUES(`{c}`)" for c in columns if c != "time")

    insert_sql = (
        f"INSERT INTO `{table}` ({col_list}) VALUES ({placeholders}) "
        f"ON DUPLICATE KEY UPDATE {update_clause}"
    )

    cursor = conn.cursor()
    # pymysql's executemany already splits these rows across several INSERT
    # statements, capped by Cursor.max_stmt_length (1 MB), so no manual
    # chunking is needed. One file is at most a day of data.
    cursor.executemany(insert_sql, df.itertuples(index=False, name=None))
    conn.commit()
    cursor.close()
    print(f"Wrote {len(df):,} rows to {table}")


# ---------------------------------------------------------------------------
# Creating / rebuilding the aggregate table
# ---------------------------------------------------------------------------
def _time_column_definition(conn) -> str:
    """
    Column definition to use for `time` in aggregated_1min, copied from
    whichever raw table exists so the aggregate matches the raw precision
    (DATETIME vs DATETIME(3), etc.) instead of hardcoding a guess.
    """
    cursor = conn.cursor()
    try:
        for table in RAW_TABLES:
            cursor.execute(
                """
                SELECT COLUMN_TYPE FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s AND COLUMN_NAME = 'time'
                """,
                (MYSQL_DATABASE, table),
            )
            row = cursor.fetchone()
            if row:
                return row[0].upper()
    finally:
        cursor.close()
    return "DATETIME"


def table_row_count(conn, table: str) -> int | None:
    """Exact row count for `table`, or None if it does not exist."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
        return int(cursor.fetchone()[0])
    except pymysql.err.MySQLError:
        return None
    finally:
        cursor.close()


def create_aggregate_table(conn, drop_existing: bool = False) -> None:
    """
    Create aggregated_1min with only its `time` primary key. Measurement
    columns are added on demand by ensure_columns().

    With drop_existing=True the table is dropped first, discarding all
    aggregates. This is safe because aggregated_1min is derived data — it can
    always be recomputed from the raw tables — and is the intended way to
    recover from a table whose columns were created with the wrong types.
    """
    time_type = _time_column_definition(conn)
    cursor = conn.cursor()
    if drop_existing:
        cursor.execute(f"DROP TABLE IF EXISTS `{AGG_TABLE}`")
        print(f"Dropped `{AGG_TABLE}`")
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `{AGG_TABLE}` (
          `time` {time_type} NOT NULL,
          PRIMARY KEY (`time`)
        ) ENGINE=InnoDB
        """
    )
    conn.commit()
    cursor.close()
    _COLUMN_TYPES.pop(AGG_TABLE, None)
    print(f"Created `{AGG_TABLE}` (`time` {time_type} PRIMARY KEY)")


def confirm_rebuild(conn, first_day: date, last_day: date,
                    record: tuple[date | None, date | None],
                    assume_yes: bool) -> bool:
    """Ask before dropping aggregated_1min; returns True to proceed."""
    n_rows = table_row_count(conn, AGG_TABLE)
    record_start, record_end = record

    if n_rows is None:
        print(f"`{AGG_TABLE}` does not exist yet; it will be created.")
        return True

    print(f"\nAbout to DROP `{AGG_TABLE}` ({n_rows:,} rows) and recompute "
          f"{first_day} → {last_day} from the raw tables.")
    print(f"  The logbook flag columns ({', '.join(FLAG_COLUMNS)}) go with it; "
          f"re-run --mode logbook afterwards to restore them.")
    if record_start is not None and (first_day > record_start or last_day < record_end):
        print(f"  WARNING: the raw record covers {record_start} → {record_end}. "
              f"Aggregates outside the requested range will be dropped and NOT "
              f"recomputed. Omit --start-date/--end-date to rebuild everything.")
    if assume_yes:
        return True
    return input("Type 'yes' to continue: ").strip().lower() == "yes"


# ---------------------------------------------------------------------------
# Per-day aggregation using data queried back from MySQL
# ---------------------------------------------------------------------------
def read_raw_table(conn, table: str, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Query one day of raw data from MySQL and return a DatetimeIndex DataFrame.

    Every measurement column is forced to float64. Without this, any column
    that is entirely NULL over the queried window comes back as object dtype,
    which breaks numeric aggregation downstream.
    """
    query = (
        f"SELECT * FROM `{table}` "
        f"WHERE `time` >= %s AND `time` < %s "
        f"ORDER BY `time`"
    )
    df = pd.read_sql(query, conn, params=(start_str, end_str))
    if df is None or df.empty:
        return pd.DataFrame()

    measurements = [c for c in df.columns if c != "time"]
    df[measurements] = (df[measurements]
                        .apply(pd.to_numeric, errors="coerce")
                        .astype("float64"))
    # Applied on read as well as on parse, so known-bad values already stored
    # in the raw tables cannot reach the aggregates.
    df = apply_pattern_nan_values(df, PATTERN_NAN_VALUES)

    df["time"] = pd.to_datetime(df["time"])
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
        if df.empty:
            continue
        df = df[~df.duplicated(keep="first")]
        if df.index.duplicated().any():
            print(f"Warning: {table} still contains duplicated indexes "
                  f"after dropping duplicates!")
            df = df[~df.index.duplicated(keep="first")]
        if not df.empty:
            raw_frames[table] = df

    if not raw_frames:
        return

    data_1min = aggregate_1min(raw_frames)
    solar_parameters = get_solar_parameters(
        data_1min.index, latitude=LATITUDE, longitude=LONGITUDE, altitude=ALTITUDE)
    agg = pd.concat([data_1min, solar_parameters], axis=1, sort=False)
    if agg.index.tz is not None:
        agg = agg.tz_localize(None)
    write_dataframe(conn, agg, AGG_TABLE)


def raw_data_range(conn) -> tuple[date | None, date | None]:
    """Earliest and latest timestamp present across all raw tables."""
    cursor = conn.cursor()
    lo, hi = None, None
    for table in RAW_TABLES:
        cursor.execute(f"SELECT MIN(`time`), MAX(`time`) FROM `{table}`")
        row_min, row_max = cursor.fetchone()
        if row_min is not None and (lo is None or row_min < lo):
            lo = row_min
        if row_max is not None and (hi is None or row_max > hi):
            hi = row_max
    cursor.close()
    return (lo.date() if lo else None, hi.date() if hi else None)


def aggregate_range(conn, first_day: date, last_day: date) -> None:
    """Recompute 1-minute aggregates for every day in [first_day, last_day]."""
    n_days = (last_day - first_day).days + 1
    print(f"\nComputing 1-minute aggregates from {first_day} to {last_day} "
          f"({n_days} days)...")
    day = first_day
    while day <= last_day:
        print(f"  Aggregating {day}...", end="\r")
        aggregate_day(conn, day)
        day += timedelta(days=1)
    print(f"\nAggregation complete ({n_days} days)")


# ---------------------------------------------------------------------------
# Logbook flags
# ---------------------------------------------------------------------------
def read_logbook(url: str | None = None) -> pd.DataFrame:
    """
    Read the station logbook and add UTC `start_time` / `end_time` columns.

    `url` defaults to LOGBOOK_URL but is resolved at call time, so a local
    .xlsx copy can be passed in directly (handy for tests and offline work).

    Everything is read as string and parsed here, because the sheet mixes date
    formats and has free-text columns that pandas would otherwise type-guess.
    """
    logbook = pd.read_excel(url or LOGBOOK_URL, dtype=str)

    for prefix, column in (("Start", "start_time"), ("End", "end_time")):
        stamp = (logbook[f"{prefix} date"].str.strip() + " "
                 + logbook[f"{prefix} time"].str.strip())
        logbook[column] = (
            pd.to_datetime(stamp, format="mixed", errors="coerce")
            # ambiguous=True resolves the repeated hour in October to the DST
            # (summer time) reading; nonexistent times in March are pushed
            # forward. Both windows are at night, so no irradiance is affected.
            .dt.tz_localize(LOGBOOK_TIMEZONE, ambiguous=True,
                            nonexistent="shift_forward")
            .dt.tz_convert(None)
        )

    return logbook


def compute_logbook_flags(
    logbook: pd.DataFrame,
    record_start: pd.Timestamp = LOGBOOK_RECORD_START,
) -> pd.DataFrame:
    """
    Derive 1-minute boolean flags from the logbook.

    Columns:
      cleaning_flag      True while a cleaning / inspection was in progress.
      soiling_flag_{c}   True over the period leading up to an observation of
                         soiling level 3 or 4 on component c, going back to the
                         previous cleaning or CLEANING_DAYS_THRESHOLD days,
                         whichever is shorter.

    The index is minute-aligned UTC and spans the logbook's own extent. False
    therefore means "logged as clean", not "no information": outside this span
    no rows are written at all, so a join against aggregated_1min gives NULL.
    """
    start_time = max(logbook["start_time"].min(), record_start).floor("1min")
    end_time = logbook["end_time"].max().ceil("1min")
    times = pd.date_range(start_time, end_time, freq="1min")
    flags = pd.DataFrame(index=times)

    # Cleaning flag — one interval per cleaning event.
    cleaning_events = logbook[logbook["Event type"] == CLEANING_EVENT_TYPE]
    flags["cleaning_flag"] = False
    for _, row in cleaning_events.iterrows():
        cleaning_period = ((flags.index >= row["start_time"])
                           & (flags.index <= row["end_time"]))
        flags.loc[cleaning_period, "cleaning_flag"] = True

    # Soiling flags — one lookback window per severe soiling observation.
    for component in SOILING_COMPONENTS:
        soiling_column = f"Soiling level (before cleaning) [{component.upper()}]"
        soiling_events = logbook[logbook[soiling_column].isin(SOILING_LEVELS_TO_FLAG)]

        column = f"soiling_flag_{component}"
        flags[column] = False
        for _, row in soiling_events.iterrows():
            soiling_start = row["start_time"]

            # Identify whether there was a cleaning event in the lookback window
            prev_cleaning_times = flags.index[
                (flags.index
                 >= soiling_start - pd.Timedelta(days=CLEANING_DAYS_THRESHOLD))
                & (flags.index < soiling_start)
                & flags["cleaning_flag"]
            ]

            if not prev_cleaning_times.empty:
                # if there was a cleaning event, flag up to the event
                period_start = prev_cleaning_times.max()
            else:
                # if there was no cleaning event, flag the whole lookback window
                period_start = (soiling_start
                                - pd.Timedelta(days=CLEANING_DAYS_THRESHOLD))

            mask = (flags.index >= period_start) & (flags.index <= soiling_start)
            flags.loc[mask, column] = True

    return flags


def reset_flag_columns(conn) -> None:
    """
    DROP the logbook flag columns from aggregated_1min so the next write
    recreates them from scratch.

    Necessary because the flags are derived, not accumulated: if an event's
    start or end time moves, or an entry is deleted, the minutes it used to
    cover fall outside the newly derived range and an upsert would never
    revisit them. Dropping the columns guarantees the table holds exactly what
    the current logbook says, and NULL everywhere else.

    All the drops go in one ALTER so the table is touched once. INSTANT is
    tried first (MySQL 8.0.29 and later can drop a column without rewriting
    the table); if the server refuses, it picks the algorithm itself.
    """
    present = [c for c in FLAG_COLUMNS if c in get_column_types(conn, AGG_TABLE)]
    if not present:
        return

    drops = ", ".join(f"DROP COLUMN `{c}`" for c in present)
    cursor = conn.cursor()
    try:
        cursor.execute(f"ALTER TABLE `{AGG_TABLE}` {drops}, ALGORITHM=INSTANT")
    except pymysql.err.MySQLError:
        print("  (server cannot drop columns instantly; rewriting the table)")
        cursor.execute(f"ALTER TABLE `{AGG_TABLE}` {drops}")
    conn.commit()
    cursor.close()
    _COLUMN_TYPES.pop(AGG_TABLE, None)
    print(f"  Dropped flag column(s): {', '.join(present)}")


def run_logbook_ingest(conn, start_date: date | None = None,
                       end_date: date | None = None,
                       url: str | None = None) -> None:
    """
    Read the logbook, derive 1-minute flags and write them into aggregated_1min.

    Flags are always derived over the logbook's full extent — a soiling window
    reaches back up to CLEANING_DAYS_THRESHOLD days, so computing only a slice
    would truncate windows at its edge. --start-date/--end-date restrict what
    is written, not what is computed.

    Unless a date range is given, the flag columns are dropped and recreated,
    so the result reflects the current logbook exactly with nothing left over
    from a previous run.
    """
    url = url or LOGBOOK_URL
    print(f"Reading logbook from {url}")
    logbook = read_logbook(url)
    print(f"  {len(logbook):,} entries "
          f"({logbook['start_time'].min()} → {logbook['end_time'].max()} UTC)")

    flags = compute_logbook_flags(logbook)
    print(f"  Derived {len(flags.columns)} flag(s) over "
          f"{flags.index.min()} → {flags.index.max()} "
          f"({len(flags):,} minutes, UTC)")
    for col in flags.columns:
        flagged = int(flags[col].sum())
        share = 100 * flagged / len(flags) if len(flags) else 0.0
        print(f"    {col}: {flagged:,} minutes flagged ({share:.1f}%)")

    flags = clip_to_range(flags, start_date, end_date)
    if flags.empty:
        print("\nNo flags fall inside the requested range — nothing written")
        return

    create_aggregate_table(conn)  # no-op if it already exists

    if start_date is None and end_date is None:
        reset_flag_columns(conn)
    else:
        print("  Date range given, so the columns are updated in place; "
              "re-run without --start-date/--end-date to rebuild them.")

    # Minutes outside the logbook's extent are never written and stay NULL,
    # keeping "no information" distinct from "logged as clean".
    write_dataframe(conn, flags.astype("uint8"), AGG_TABLE)


# ---------------------------------------------------------------------------
# FTP download
# ---------------------------------------------------------------------------
def get_ftp_connection() -> ftplib.FTP:
    """Open and authenticate a plain FTP connection to the SARC FTP server."""
    ftp = ftplib.FTP()
    ftp.connect(FTP_HOST, FTP_PORT)
    ftp.login(FTP_USER, FTP_PASSWORD)
    return ftp


def download_file(ftp: ftplib.FTP, filename: str) -> str | None:
    """Download a single file from FTP, return content as string or None."""
    remote_path = f"{FTP_DIR.rstrip('/')}/{filename}"
    buf = io.BytesIO()
    try:
        ftp.retrbinary(f"RETR {remote_path}", buf.write)
        return buf.getvalue().decode("utf-8", errors="replace")
    except ftplib.error_perm as e:
        print(f"  WARNING: Could not download {remote_path}: {e}")
        return None


def fetch_ftp_table_day(ftp: ftplib.FTP, table: str, day: date) -> pd.DataFrame:
    """Download and parse one table's daily file for a given calendar day."""
    filename = FILE_PATTERNS[table].format(date=day.strftime("%Y-%m-%d"))
    content = download_file(ftp, filename)
    if content is None:
        return pd.DataFrame()
    return parse_toa5(content, source=filename)


# ---------------------------------------------------------------------------
# Shared ingestion helpers
# ---------------------------------------------------------------------------
def ingest_dataframe(conn, df: pd.DataFrame, table: str, overall_range: tuple) -> tuple:
    """Write df to `table` and fold its time span into (overall_start, overall_end)."""
    overall_start, overall_end = overall_range
    if df.empty:
        return overall_start, overall_end

    # Remove timezone information to avoid PyMySQL modifying the timezone
    if df.index.tz is not None:
        df = df.tz_convert("UTC")
        df = df.tz_localize(None)
    write_dataframe(conn, df, table)

    if overall_start is None or df.index.min() < overall_start:
        overall_start = df.index.min()
    if overall_end is None or df.index.max() > overall_end:
        overall_end = df.index.max()
    return overall_start, overall_end


def clip_to_range(df: pd.DataFrame, start_date: date | None,
                  end_date: date | None) -> pd.DataFrame:
    """Keep rows from start_date 00:00 up to the end of end_date, inclusive."""
    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index < pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    return df


def run_sd_ingest(conn, source_dir: str, start_date: date | None = None,
                  end_date: date | None = None) -> tuple:
    """
    Ingest SD card .dat files under source_dir, optionally restricted to
    [start_date, end_date] (both inclusive, both optional).
    Returns (start, end) of the data actually written.
    """
    print(f"Scanning {source_dir} for SD card .dat files...")
    if start_date or end_date:
        print(f"Restricting to {start_date or 'start of record'} → "
              f"{end_date or 'end of record'}")
    files_by_table = find_sd_files(source_dir, end_date=end_date)

    overall_start, overall_end = None, None
    for table in RAW_TABLES:
        files = files_by_table.get(table, [])
        if not files:
            print(f"\n{table}: no files found, skipping")
            continue
        print(f"\n{table}: ingesting {len(files)} files")
        for path in files:
            print(f"  Reading {path.name}...")
            content = path.read_text(encoding="utf-8", errors="replace")
            df = parse_toa5(content, source=path.name)
            if df.empty:
                print("    (empty after parsing, skipping)")
                continue
            n_parsed = len(df)
            df = clip_to_range(df, start_date, end_date)
            if df.empty:
                print(f"    (all {n_parsed:,} rows outside the date range, "
                      f"skipping)")
                continue
            if len(df) != n_parsed:
                print(f"    Kept {len(df):,}/{n_parsed:,} rows in range")
            print(f"    Parsed {len(df):,} rows  "
                  f"({df.index.min()} → {df.index.max()})")
            overall_start, overall_end = ingest_dataframe(
                conn, df, table, (overall_start, overall_end))

    return overall_start, overall_end


def run_ftp_ingest(conn, start_date: date, end_date: date, tables: list[str]) -> tuple:
    """
    Ingest daily FTP files for `tables` over [start_date, end_date] (inclusive).
    Missing files (e.g. a day the logger didn't publish) are skipped with a warning.
    """
    if not (FTP_HOST and FTP_USER and FTP_PASSWORD):
        raise SystemExit(
            "FTP mode requires SARC_FTP_HOST, SARC_FTP_USER and SARC_FTP_PASSWORD "
            "to be set as environment variables."
        )

    print(f"Connecting to FTP server {FTP_HOST}...")
    ftp = get_ftp_connection()

    overall_start, overall_end = None, None
    try:
        for table in tables:
            print(f"\n{table}: fetching {start_date} → {end_date}")
            day = start_date
            while day <= end_date:
                print(f"  {day}: downloading "
                      f"{FILE_PATTERNS[table].format(date=day)}...")
                df = fetch_ftp_table_day(ftp, table, day)
                if df.empty:
                    print(f"    (no data for {day}, skipping)")
                else:
                    print(f"    Parsed {len(df):,} rows")
                    overall_start, overall_end = ingest_dataframe(
                        conn, df, table, (overall_start, overall_end))
                day += timedelta(days=1)
    finally:
        try:
            ftp.quit()  # graceful QUIT so the server frees the slot right away
        except Exception:
            ftp.close()

    return overall_start, overall_end


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}, expected YYYY-MM-DD")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest SARC data into MySQL from SD card files or the daily "
                    "FTP feed, and compute 1-minute aggregates"
    )
    parser.add_argument(
        "--mode",
        choices=["sd", "ftp", "reaggregate", "logbook", "schema"],
        required=True,
        help="sd: backfill from SD card .dat files. "
             "ftp: fetch daily .csv files for a date range. "
             "reaggregate: recompute 1-min aggregates from raw data already in "
             "MySQL. logbook: derive 1-min quality flags from the logbook "
             "spreadsheet. schema: report columns with unexpected types.",
    )
    parser.add_argument(
        "--source",
        default=SD_DATA_DIR,
        help=f"[sd mode] Directory containing Station_SD_data files "
             f"(default: {SD_DATA_DIR})",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        help="First day to process, inclusive (YYYY-MM-DD). "
             "[sd mode] Rows before this are skipped. Default: no lower bound. "
             "[ftp mode] First day to fetch. Default: yesterday. "
             "[reaggregate mode] First day to recompute. Default: start of record. "
             "[logbook mode] First day to write. Default: start of the logbook.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        help="Last day to process, inclusive (YYYY-MM-DD). "
             "[sd mode] Rows after this are skipped. Default: no upper bound. "
             "[ftp mode] Default: --start-date, or yesterday if that is omitted. "
             "[reaggregate mode] Last day to recompute. Default: end of record. "
             "[logbook mode] Last day to write. Default: end of the logbook.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=RAW_TABLES,
        default=RAW_TABLES,
        help="[ftp mode] Which raw tables to fetch (default: all four). "
             "Useful for filling holes in a single dataset over a date range.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="[reaggregate mode] DROP and recreate aggregated_1min before "
             "recomputing. Use this to recover from a table whose columns were "
             "created with the wrong types. Raw tables are never touched, but "
             "the logbook flag columns are dropped too — re-run --mode logbook "
             "afterwards.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt for --rebuild (for scheduled runs).",
    )
    parser.add_argument(
        "--skip-aggregation",
        action="store_true",
        help="Ingest raw data only; do not compute 1-minute aggregates.",
    )
    args = parser.parse_args()

    if args.rebuild and args.mode != "reaggregate":
        parser.error("--rebuild is only valid with --mode reaggregate")

    if (args.start_date and args.end_date and args.start_date > args.end_date):
        parser.error("--start-date must not be after --end-date")

    try:
        conn = get_connection()
    except pymysql.err.MySQLError as err:
        errno = err.args[0] if err.args else None
        if errno == ER.ACCESS_DENIED_ERROR:
            print("MySQL access denied — check MYSQL_SARC_USER / MYSQL_SARC_PASSWORD")
        elif errno == ER.BAD_DB_ERROR:
            print(f"Database '{MYSQL_DATABASE}' does not exist")
        else:
            print(err)
        return

    try:
        if args.mode == "schema":
            check_schema(conn)
            return

        if args.mode == "reaggregate":
            record = raw_data_range(conn)
            record_start, record_end = record
            if record_start is None:
                print("No raw data found — nothing to aggregate")
                return
            first_day = args.start_date or record_start
            last_day = args.end_date or record_end

            if args.rebuild:
                if not confirm_rebuild(conn, first_day, last_day, record, args.yes):
                    print("Aborted; nothing was changed.")
                    return
                create_aggregate_table(conn, drop_existing=True)

            aggregate_range(conn, first_day, last_day)
            print("Done.")
            return

        if args.mode == "logbook":
            run_logbook_ingest(
                conn,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            print("Done.")
            return

        if args.mode == "sd":
            overall_start, overall_end = run_sd_ingest(
                conn, args.source, args.start_date, args.end_date)
        else:
            end_date = (args.end_date or args.start_date
                        or (date.today() - timedelta(days=1)))
            start_date = args.start_date or end_date
            overall_start, overall_end = run_ftp_ingest(
                conn, start_date, end_date, args.tables)

        if overall_start is None:
            print("\nNo data ingested — skipping aggregation")
            return

        if args.skip_aggregation:
            print("\n--skip-aggregation given; not computing 1-minute aggregates")
            return

        create_aggregate_table(conn)  # no-op if it already exists
        aggregate_range(conn, overall_start.date(), overall_end.date())
        print("Done.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()