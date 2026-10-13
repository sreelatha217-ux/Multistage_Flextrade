# Read-only inspection of the PostgreSQL database + DK1 day-ahead coverage report.
# Never prints the connection string or password.
# Usage: export DATABASE_URL='postgres://...'; uv run --with psycopg2-binary python inspect_db.py 2>&1 | tee db_schema.txt
import os
import sys

url = os.environ.get("DATABASE_URL")
if not url:
    sys.exit("Set DATABASE_URL first (Aiven console -> service overview -> Service URI).")

import psycopg2

conn = psycopg2.connect(url)
conn.set_session(readonly=True, autocommit=True)
cur = conn.cursor()

# Skip TimescaleDB internal schemas so only the real tables are listed
cur.execute(
    "select table_schema, table_name from information_schema.tables "
    "where table_schema not in ('pg_catalog','information_schema') "
    "and table_schema not like '\\_timescaledb%' "
    "and table_type in ('BASE TABLE','VIEW') order by 1, 2"
)
tables = cur.fetchall()
print(f"{len(tables)} tables/views found\n")

for schema, name in tables[:60]:
    full = f'"{schema}"."{name}"'
    print("=" * 90)
    print(full)
    cur.execute(
        "select column_name, data_type from information_schema.columns "
        "where table_schema=%s and table_name=%s order by ordinal_position",
        (schema, name),
    )
    cols = cur.fetchall()
    try:
        cur.execute(f"select count(*) from {full}")
        print("rows:", cur.fetchone()[0])
    except Exception as err:
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
    except Exception as err:
        print("  sample rows: n/a", str(err).splitlines()[0])

print("\n" + "=" * 90)
print("DK1 day-ahead coverage")
try:
    cur.execute(
        "select zone, source, count(*), min(time_utc), max(time_utc) "
        "from public.day_ahead_prices group by 1, 2 order by 1, 2"
    )
    print("zone / source / rows / first / last:")
    for r in cur.fetchall():
        print("  ", r)

    cur.execute(
        "with d as ("
        " select (time_dk at time zone 'Europe/Copenhagen')::date as day,"
        " avg(price_eur) as mean_p, min(price_eur) as min_p, max(price_eur) as max_p, count(*) as n"
        " from public.day_ahead_prices where zone='DK1' and source='dayahead_real' group by 1)"
        " select to_char(day,'YYYY-MM') as month, count(*) as days,"
        " count(*) filter (where n in (92,96,100)) as full_days,"
        " round(avg(mean_p)::numeric,1) as mean_price,"
        " count(*) filter (where min_p < 0) as neg_days,"
        " round(max(max_p)::numeric,0) as peak"
        " from d group by 1 order by 1"
    )
    print("\nmonth / days / full_days / mean EUR/MWh / neg_days / peak:")
    for r in cur.fetchall():
        print("  ", r)
except Exception as err:
    print("coverage query failed:", str(err).splitlines()[0])

print("\ndone")
