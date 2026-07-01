from datetime import date, timedelta
import subprocess
import sys

start = date(2026, 6, 10)
end = date.today()

current = start
while current <= end:
    date_str = current.strftime("%Y-%m-%d")
    print(f"Processing {date_str}...")
    result = subprocess.run(
        [sys.executable, r"C:\github\sarc\influxdb\sarc_ingest.py", "--date", date_str],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"  WARNING: {date_str} failed with return code {result.returncode}")
    current += timedelta(days=1)

print("Backfill complete.")
