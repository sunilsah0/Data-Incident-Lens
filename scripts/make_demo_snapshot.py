import os
import shutil

import duckdb

SRC, DST = "data/metadata.duckdb", "demo/metadata_snapshot.duckdb"
os.makedirs("demo", exist_ok=True)
con = duckdb.connect(SRC)
con.execute("CHECKPOINT")
con.close()
shutil.copy(SRC, DST)
print(f"Snapshot saved: {os.path.getsize(DST) / 1e6:.1f} MB")