import pvlib
from influxdb_client_3 import InfluxDBClient3
import os


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


start = "2025-04-01 00:00"
end = "2026-06-01 00:00"

clearsky = get_mcclear(
    latitude=55.79064,
    longitude=12.52505,
    start=start,
    end=end,
)

api_token = os.environ["INFLUXDB3_AUTH_TOKEN"]
sarc_host = os.environ["INFLUXDB3_HOST"]

client = InfluxDBClient3(
    host=sarc_host,
    database="sarc",
    token=api_token,
)

write_influxdb3_data(
    client,
    df=clearsky,
    table="aggregated_1min",
    chunk_size="1ME",
)
