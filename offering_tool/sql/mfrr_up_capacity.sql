-- REPLACE THIS FILE WITH YOUR OWN mFRR QUERY. Keep the three output columns and the three parameters.
--
-- Output columns (exactly these names):
--   delivery_hour_start              timestamp WITH time zone
--   area                             text, e.g. 'DK1'
--   mfrr_up_capacity_price_eur_mw_h  numeric, mFRR UP capacity price in EUR/MW/h (not the down direction)
-- Rules: one row per hour (average 15-minute rows here). EUR, not DKK.
-- Parameters: %(area)s, %(first_day)s, %(last_day)s (same as the DA query)
--
-- Placeholder, table and column names are made up:
select date_trunc('hour', ts)  as delivery_hour_start,
       area,
       avg(up_price_eur_mw_h)  as mfrr_up_capacity_price_eur_mw_h
from   mfrr_capacity
where  area = %(area)s
  and  ts >= (%(first_day)s::timestamp at time zone 'Europe/Copenhagen')
  and  ts <  ((%(last_day)s::date + 1)::timestamp at time zone 'Europe/Copenhagen')
group  by 1, 2
order  by 1;
