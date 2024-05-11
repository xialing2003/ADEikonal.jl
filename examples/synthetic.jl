using CSV
using DataFrames
using HDF5
using PyCall
using ADCME
using ADTomo
using Random
using JSON
using Dates
Random.seed!(123)

## define research region
config = Dict(
           "minlatitude" => 30.0,
           "maxlatitude" => 31.0,
           "minlongitude" => 130.0,
           "maxlongitude" => 131.0,
           "mindepth" => 0.0,
           "maxdepth" => 20.0,
           "degree2km" => 111.19
       )

time0 = DateTime("2019-01-01T00:00:00", "yyyy-mm-ddTHH:MM:SS")
config["depth0"] = (config["mindepth"] + config["maxdepth"]) / 2
config["latitude0"] = (config["minlatitude"] + config["maxlatitude"]) / 2
config["longitude0"] = (config["minlongitude"] + config["maxlongitude"]) / 2

open("config.json", "w") do f
    n = length(config); i = 0
    println(f, "{")
    for (key, value) in config
        i += 1
        if i < n println(f, "\"$key\": $value,")
        else println(f, "\"$key\": $value")
        end
    end
    println(f, "}")
end

## build stations

num_station = 50
stations = DataFrame(station_id=[], longitude=[], latitude=[], elevation_m=[], x_km = [], y_km = [], z_km = [])

for i in 1:num_station
    station_id = "NC.$(lpad(i, 2, '0'))"
    latitude = config["latitude0"] + (rand() - 0.5) * 1
    longitude = config["longitude0"] + (rand() - 0.5) * 1
    elevation_m = rand() * 1000
    depth_km = -elevation_m / 1000
    push!(stations, (station_id=station_id, longitude=longitude, latitude=latitude, elevation_m=elevation_m, x_km=missing, y_km=missing, z_km=depth_km))
end

pyproj = pyimport("pyproj") # use it in an environment with pyproj
proj = pyproj.Proj(
    "+proj=sterea +lon_0=$( (config["minlongitude"] + config["maxlongitude"]) / 2 ) +lat_0=$( (config["minlatitude"] + config["maxlatitude"]) / 2 ) +units=km"
)
stations.x_km, stations.y_km = proj(stations.longitude, stations.latitude)

CSV.write("stations.csv", stations)

## build events

num_event = 200
events = DataFrame(event_index=[], longitude=[], latitude=[], magnitude=[], x_km = [], y_km = [], z_km = [])
index = 0
for i in 1:num_event
    global index += 1 
    latitude = config["latitude0"] + (rand() - 0.5) * 1
    longitude = config["longitude0"] + (rand() - 0.5) * 1
    magnitude = rand()*2
    depth_km = config["depth0"] + (rand()-0.5) * 20
    push!(events, (event_index=index, longitude=longitude, latitude=latitude, magnitude=magnitude, x_km=missing, y_km=missing, z_km=depth_km))
end
events.x_km, events.y_km = proj(events.longitude, events.latitude)

CSV.write("events.csv", events)

## adjust locations for calculation

h = 1.0; dx, dy = proj(config["minlongitude"],config["minlatitude"])
dx = ceil(abs(dx)); stations.x = (stations.x_km .+ dx) ./ h .+ 1; events.x = (events.x_km .+ dx) ./ h .+ 1
dy = ceil(abs(dy)); stations.y = (stations.y_km .+ dy) ./ h .+ 1; events.y = (events.y_km .+ dy) ./ h .+ 1
stations.z = (stations.z_km .+ 1) ./ h .+ 1; events.z = events.z_km ./ h .+ 1

m, n = proj(config["maxlongitude"], config["maxlatitude"])
m = convert(Int64,ceil((m + dx)/h)) + 1; n = convert(Int64, ceil((n + dy)/h)) + 1
l = convert(Int64, ceil(config["maxdepth"]/h)) + 1

## develop data of picks

vel_p = ones(m,n,l) * 6.0; vel_s = ones(m,n,l) * 3.5 # (98, 113, 21)
vel_p[40:60, 50:70, 8:14] .= 6.5; vel_s[40:60, 50:70, 8:14] .= 3.8

up = PyObject[]; fvel_p = 1 ./ vel_p; fvel_s = 1 ./ vel_s
for i = 1:num_station
    ix = stations.x[i]; ixu = convert(Int64, ceil(ix)); ixd = convert(Int64, floor(ix))
    iy = stations.y[i]; iyu = convert(Int64, ceil(iy)); iyd = convert(Int64, floor(iy))
    iz = stations.z[i]; izu = convert(Int64, ceil(iz)); izd = convert(Int64, floor(iz))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*fvel_p[ixu,iyu,izu]*h
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*fvel_p[ixu,iyu,izd]*h
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*fvel_p[ixu,iyd,izu]*h
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*fvel_p[ixu,iyd,izd]*h
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*fvel_p[ixd,iyu,izu]*h
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*fvel_p[ixd,iyu,izd]*h
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*fvel_p[ixd,iyd,izu]*h
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*fvel_p[ixd,iyd,izd]*h
    push!(up, eikonal3d(u0, fvel_p, h, m, n, l, 1e-3, false))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*fvel_s[ixu,iyu,izu]*h
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*fvel_s[ixu,iyu,izd]*h
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*fvel_s[ixu,iyd,izu]*h
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*fvel_s[ixu,iyd,izd]*h
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*fvel_s[ixd,iyu,izu]*h
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*fvel_s[ixd,iyu,izd]*h
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*fvel_s[ixd,iyd,izu]*h
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*fvel_s[ixd,iyd,izd]*h
    push!(us, eikonal3d(u0, fvel_s, h, m, n, l, 1e-3, false))
end
sess = Session(); init(sess)
utravel_p = run(sess,up); utravel_s = run(sess,us)
scaltime_p = ones(numsta,numeve); qua_p = ones(numsta,numeve)
scaltime_s = ones(numsta,numeve); qua_s = ones(numsta,numeve)
for i = 1:numsta
    for j = 1:numeve
        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = x1 + 1
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = y1 + 1
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = z1 + 1
        # P wave
        txyz = (z2-jz)*(y2-jy)*((x2-jx)*ubeg_p[i][x1,y1,z1] + (jx-x1)*ubeg_p[i][x2,y1,z1]) +
               (z2-jz)*(jy-y1)*((x2-jx)*ubeg_p[i][x1,y2,z1] + (jx-x1)*ubeg_p[i][x2,y2,z1]) + 
               (jz-z1)*(y2-jy)*((x2-jx)*ubeg_p[i][x1,y1,z2] + (jx-x1)*ubeg_p[i][x2,y1,z2]) + 
               (jz-z1)*(jy-y1)*((x2-jx)*ubeg_p[i][x1,y2,z2] + (jx-x1)*ubeg_p[i][x2,y2,z2])
        scaltime_p[i,j] = txyz
        qua_p[i,j] = rand()/2 + 0.5
        # S wave
        txyz = (z2-jz)*(y2-jy)*((x2-jx)*ubeg_s[i][x1,y1,z1] + (jx-x1)*ubeg_s[i][x2,y1,z1]) +
               (z2-jz)*(jy-y1)*((x2-jx)*ubeg_s[i][x1,y2,z1] + (jx-x1)*ubeg_s[i][x2,y2,z1]) + 
               (jz-z1)*(y2-jy)*((x2-jx)*ubeg_s[i][x1,y1,z2] + (jx-x1)*ubeg_s[i][x2,y1,z2]) + 
               (jz-z1)*(jy-y1)*((x2-jx)*ubeg_s[i][x1,y2,z2] + (jx-x1)*ubeg_s[i][x2,y2,z2])
        scaltime_s[i,j] = txyz
        qua_s[i,j]  = rand()/2 + 0.5
    end
end



# scaltime_d = scaltime_s - scaltime_p

# if isfile(folder * "for_P/uobs_p.h5")
#     rm(folder * "for_P/uobs_p.h5")
#     rm(folder * "for_P/qua_p.h5")
#     rm(folder * "for_S/uobs_s.h5")
#     rm(folder * "for_S/qua_s.h5")
#     rm(folder * "uobs_d.h5")
# end
# h5write(folder * "for_P/uobs_p.h5","matrix",scaltime_p)
# h5write(folder * "for_S/uobs_s.h5","matrix",scaltime_s)
# h5write(folder * "for_P/qua_p.h5","matrix",qua_p)
# h5write(folder * "for_S/qua_s.h5","matrix",qua_s)
# h5write(folder * "uobs_d.h5","matrix",scaltime_d)

# # during inversion, vp can change in [-1,1], pvs can change in [-0.3,0.3]
# # I suppose that Vp/Vs ratio almost equals an constant, and varies in some regions
# # so Vp/Vs starts with this constant value, and I add a regression in the loss function
# # just like L1 norm
# # 
