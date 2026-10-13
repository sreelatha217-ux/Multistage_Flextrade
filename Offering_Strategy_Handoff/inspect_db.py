"""Read-only inspection of the PostgreSQL database: tables, columns, row counts, date ranges, sample rows.
It never prints the connection string or password.

Usage (in the codespace):
  export DATABASE_URL='postgres://USER:PASSWORD@HOST:PORT/DBNAME?sslmode=require'   # copy the Service URI from Aiven
  uv run --with psycopg2-binary python inspect_db.py 2>&1 | tee db_schema.txt
Paste db_schema.txt here (it contains no credentials).
"""
import os
import sys

url = os.environ.get("DATABASE_URL")
if not url:
    sys.exit("Set DATABASE_URL first (Aiven console -> service overview -> Service URI).")

import psycopg2

conn = psycopg2.connect(url)
conn.set_session(readonly=True, autocommit=True)
cur = conn.cursor()

cur.execute("""select table_schema, table_name from information_schema.tables
               where table_schema not in ('pg_catalog','information_schema') and table_type in ('BASE TABLE','VIEW')
               order by 1, 2""")
tables = cur.fetchall()
print(f"{len(tables)} tables/views found\n")

for schema, name in tables[:60]:
    full = f'"{schema}"."{name}"'
    print("=" * 90)
    print(full)
    cur.execute("""select column_name, data_type from information_schema.columns
                   where table_schema=%s and table_name=%s order by ordinal_position""", (schema, name))
    cols = cur.fetchall()
    try:
        cur.execute(f"select count(*) from {full}")
        print("rows:", cur.fetchone()[0])
    except Exception as err:  # noqa: BLE001
        print("rows: n/a", str(err).splitlines()[0])
        continue
    for c, t in cols:
        extra = ""
        if t.startswith("timestamp") or t == "date":
            cur.execute(f'select min("{c}"), max("{c}") from {full}')
            lo, hi = cur.fetchone()
            extra = f"   range: {lo} .. {hi}"
        print(f"  {c:32s}{t:28s}{extra}")
    try:
        cur.execute(f"select * from {full} limit 3")
        print("  sample rows:")
        for row in cur.fetchall():
            print("   ", tuple(str(x)[:40] for x in row))
    except Exception as err:  # noqa: BLE001
        print("  sample rows: n/a", str(err).splitlines()[0])
print("\ndone")
