import sys

import duckdb

name = sys.argv[1] if len(sys.argv) > 1 else None
con = duckdb.connect("data/metadata.duckdb", read_only=True)
sql = ("SELECT experiment, method, metric, round(value, 3) AS value, n "
       "FROM experiment_results")
params = []
if name:
    sql += " WHERE experiment = ?"
    params = [name]
print(con.execute(sql + " ORDER BY 1, 2, 3", params).df().to_string())