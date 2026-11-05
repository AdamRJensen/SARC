from pathlib import Path
import sarcingest as si
import pandas as pd
from pathlib import Path


path = Path(r"\\ait-pdfs.win.dtu.dk\Department\Cme\Section-ES\Solar future\SARC\incoming_data")

filename = "Razon_manually_extracted_data.csv"

data = pd.read_csv(path / filename, index_col=0, parse_dates=[0])

VARIABLE_MAP = {
    "ghi": "Razon_190116_GHI_Wm2",
    "dhi": "PR1_190116_DHI_Wm2",
    "dni": "PH1_190116_DNI_Wm2",
    "TempDiffuse (Degrees celcius)": "PR1_190116_temperature_degC",
    "TempDirect (Degrees celcius)": "PH1_190116_temperature_degC",
}

data = data.rename(columns=VARIABLE_MAP)

if data.index.tz is not None:
    data = data.tz_convert("UTC").tz_localize(None)


conn = si.get_connection()
try:
    si.write_dataframe(conn, data[VARIABLE_MAP.values()], si.AGG_TABLE)
finally:
    conn.close()
