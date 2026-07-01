import pvlib
import os
import pandas as pd


def get_mcclear(latitude, longitude, start, end):
    cams, _ = pvlib.iotools.get_cams(
        latitude=latitude,
        longitude=longitude,
        start=start,
        end=end,
        email="arajen@dtu.dk",
        identifier="mcclear",
        time_step="1min",
        label="left",
    )
    return cams[["ghi_clear", "dhi_clear", "dni_clear"]]


latitude = 55.79064
longitude = 12.52505

start = "2026-06-01"
end = pd.Timestamp.today()


df = get_mcclear(
    latitude=latitude,
    longitude=longitude,
    start=start,
    end=end,
)
INFLUXDB_HOST = os.environ["INFLUXDB3_HOST"]
INFLUXDB_TOKEN = os.environ["INFLUXDB3_AUTH_TOKEN"]
INFLUXDB_DB = os.environ["INFLUXDB3_DATABASE"]

client = InfluxDBClient3(
    host=INFLUXDB_HOST,
    database=INFLUXDB_DB,
    token=INFLUXDB_TOKEN,
)

write_influxdb3_data(
    client=client,
    df=df,
    table="aggregated_1min",
    chunk_size="1D",
)
