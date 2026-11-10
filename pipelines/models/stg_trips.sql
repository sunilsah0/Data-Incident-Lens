select
    VendorID as vendor_id,
    tpep_pickup_datetime as pickup_at,
    tpep_dropoff_datetime as dropoff_at,
    PULocationID as pickup_zone_id,
    DOLocationID as dropoff_zone_id,
    passenger_count,
    trip_distance,
    fare_amount,
    total_amount
from {{ source('raw', 'yellow_trips') }}
where fare_amount > 0
  and trip_distance > 0
  and tpep_dropoff_datetime > tpep_pickup_datetime
  and tpep_pickup_datetime >= '2024-01-01'
  and tpep_pickup_datetime < '2024-02-01'
  