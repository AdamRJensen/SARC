import pandas as pd

# Read logbook
xlsx_url = "https://docs.google.com/spreadsheets/d/11HGFfdIwgw48tLejh3nTuS204q5MC6n18jJW-pSLFfU/export?format=xlsx&gid=1885804395"
logbook = pd.read_excel(xlsx_url, dtype=str)

logbook["start_time"] = (
    pd.to_datetime(logbook["Start date"] + " " + logbook["Start time"], format="mixed")
    .dt.tz_localize("Europe/Copenhagen")
    .dt.tz_convert("UTC")
)

logbook["end_time"] = (
    pd.to_datetime(
        logbook["End date"] + " " + logbook["End time"],
        format="mixed",
    )
    .dt.tz_localize("Europe/Copenhagen")
    .dt.tz_convert("UTC")
)

# Initialize logbook flags DataFrame with 1-minute intervals
start_time = max([logbook["start_time"].min(), pd.Timestamp("2025-04-01", tz="UTC")])
end_time = logbook["end_time"].max()
times = pd.date_range(start_time, end_time, freq="1min")
logbook_flags = pd.DataFrame(index=times)

# Create cleaning flag
cleaning_events = logbook[logbook["Event type"] == "Cleaning / regular inspection"]

logbook_flags["cleaning_flag"] = False
for _, row in cleaning_events.iterrows():
    cleaning_period = (logbook_flags.index >= row["start_time"]) & (
        logbook_flags.index <= row["end_time"]
    )
    logbook_flags.loc[cleaning_period, "cleaning_flag"] = True

# Create soiling flags
cleaning_days_threshold = 3
remove_soiling_levels = ["3", "4"]

for component in ["ghi", "dhi", "dni"]:
    # Soiling level column name in the logbook
    soiling_column = f"Soiling level (before cleaning) [{component.upper()}]"

    # Create subset of logbook entries only containing events with soiling
    soiling_events = logbook[logbook[soiling_column].isin(remove_soiling_levels)]

    # Initialize soiling flag column
    logbook_flags[f"soiling_flag_{component}"] = False
    for _, row in soiling_events.iterrows():
        soiling_start = row["start_time"]

        # Identify if there was a cleaning event in the previous 3 days
        prev_cleaning_times = logbook_flags.index[
            (
                logbook_flags.index
                >= soiling_start - pd.Timedelta(days=cleaning_days_threshold)
            )
            & (logbook_flags.index < soiling_start)
            & logbook_flags["cleaning_flag"]
        ]

        if not prev_cleaning_times.empty:
            # if there was a cleaning event, flag up to the event
            period_start = prev_cleaning_times.max()
        else:
            # if there was no cleaning event in the past 3 days, flag the past 3 days
            period_start = soiling_start - pd.Timedelta(days=cleaning_days_threshold)

        mask = (logbook_flags.index >= period_start) & (
            logbook_flags.index <= soiling_start
        )
        logbook_flags.loc[mask, f"soiling_flag_{component}"] = True
