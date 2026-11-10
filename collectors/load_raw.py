import os
import urllib.request
import duckdb

DATA = "data"
os.makedirs(DATA, exist_ok=True)

TRIPS_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"

trips_file = f"{DATA}/yellow_2024-01.parquet"
zones_file = f"{DATA}/taxi_zones.csv"

for url, path in [(TRIPS_URL, trips_file), (ZONES_URL, zones_file)]:
    if not os.path.exists(path):
        print("Downloading", path)
        urllib.request.urlretrieve(url, path)

con = duckdb.connect(f"{DATA}/incidentlens.duckdb")
con.execute("CREATE SCHEMA IF NOT EXISTS raw")
con.execute(f"CREATE OR REPLACE TABLE raw.yellow_trips AS SELECT * FROM read_parquet('{trips_file}')")
con.execute(f"CREATE OR REPLACE TABLE raw.taxi_zones AS SELECT * FROM read_csv_auto('{zones_file}')")
print(con.execute("SELECT count(*) FROM raw.yellow_trips").fetchone())
con.close()