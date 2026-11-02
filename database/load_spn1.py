from pathlib import Path
import sarcingest as si
import pandas as pd
from pathlib import Path


path = Path(r"\\ait-pdfs.win.dtu.dk\Department\Cme\Section-ES\Solar future\SARC\incoming_data\SPN1_A270_Corrections")

filename = "A270_Corrected2.hdf"

data = pd.read_hdf(path / filename)

columns = []
for c in ["GHI", "DNI", "DHI"]:
    for ii in ["", "2"]:
        col_name = f"SPN1_A270_{c}_Corr{ii}"
        columns.append(col_name)

data.loc[(data[columns] < -5).any(axis=1), columns] = float("nan")
data.loc[(data[columns] > 2000).any(axis=1), columns] = float("nan")

if data.index.tz is not None:
    data = data.tz_convert("UTC").tz_localize(None)


conn = si.get_connection()
try:
    si.write_dataframe(conn, data[columns], si.AGG_TABLE)
finally:
    conn.close()
