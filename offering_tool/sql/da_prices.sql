-- REPLACE THIS FILE WITH YOUR OWN DA QUERY. Keep the three output columns and the three parameters.
--
-- Output columns (exactly these names):
--   delivery_hour_start  timestamp WITH time zone, start of the delivery hour
--   area                 text, e.g. 'DK1'
--   da_price_eur_mwh     numeric, day-ahead spot price in EUR/MWh
-- Rules: one row per hour. If your table has 15-minute rows, average the four rows of each hour here.
--        Convert to EUR/MWh here if the table holds DKK.
-- Parameters (psycopg named style): %(area)s, %(first_day)s, %(last_day)s  (dates, both inclusive, Danish local days)
--
-- Placeholder, table and column names are made up:
select date_trunc('hour', ts) as delivery_hour_start,
       area,
       avg(price_eur_mwh)     as da_price_eur_mwh
from   da_prices
where  area = %(area)s
  and  ts >= (%(first_day)s::timestamp at time zone 'Europe/Copenhagen')
  and  ts <  ((%(last_day)s::date + 1)::timestamp at time zone 'Europe/Copenhagen')
group  by 1, 2
order  by 1;
