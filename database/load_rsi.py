from pathlib import Path
import sarcingest as si
import pandas as pd
from pathlib import Path


path = Path(r"\\ait-pdfs.win.dtu.dk\Department\Cme\Section-ES\Solar future\SARC\incoming_data")

filename = "Dk-KoLyn_2025-04-01--2026-06-01_biasCorrectionSR300_new.xlsx"

data = pd.read_excel(path / filename, index_col="tmstamp", parse_dates=[0])
data = data[data.index.notna()]  # remove last row
data.index = data.index - pd.Timedelta(hours=1)  # shift to UTC
data.index = data.index - pd.Timedelta(minutes=1)  # shift to left-labeled timestamp

VARIABLE_MAP = {
    "ghiCor": "RSI_MT24204_GHI_Wm2",
    "dhiCor": "RSI_MT24204_DHI_Wm2",
    "dniCor": "RSI_MT24204_DNI_Wm2",
}

data = data.rename(columns=VARIABLE_MAP)

if data.index.tz is not None:
    data = data.tz_convert("UTC").tz_localize(None)


conn = si.get_connection()
try:
    si.write_dataframe(conn, data[VARIABLE_MAP.values()], si.AGG_TABLE)
finally:
    conn.close()
