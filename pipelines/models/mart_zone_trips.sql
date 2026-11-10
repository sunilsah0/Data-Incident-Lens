select
    t.pickup_zone_id,
    z.Zone as zone_name,
    z.Borough as borough,
    count(*) as trips,
    sum(t.total_amount) as revenue
from {{ ref('stg_trips') }} t
left join {{ source('raw', 'taxi_zones') }} z
    on t.pickup_zone_id = z.LocationID
group by 1, 2, 3