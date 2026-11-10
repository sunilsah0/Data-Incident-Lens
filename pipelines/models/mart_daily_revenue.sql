select
    cast(pickup_at as date) as trip_date,
    count(*) as trips,
    sum(total_amount) as revenue,
    avg(fare_amount) as avg_fare
from {{ ref('stg_trips') }}
group by 1