push!(LOAD_PATH,"../src")
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

folder = "anomaly"
# folder = "checkerboard"
# len = 10

## define research region
config = Dict{String, Any}()
config["minlatitude"] = 30.0; config["minlongitude"] = 130.0; config["mindepth"] = 0.0
config["maxlatitude"] = 30.5; config["maxlongitude"] = 130.5; config["maxdepth"] = 20.0
config["degree2km"] = 111.19

time0 = DateTime("2019-01-01T00:00:00", "yyyy-mm-ddTHH:MM:SS")
config["depth0"] = (config["mindepth"] + config["maxdepth"]) / 2
config["latitude0"] = (config["minlatitude"] + config["maxlatitude"]) / 2
config["longitude0"] = (config["minlongitude"] + config["maxlongitude"]) / 2
config["latituderange"] = config["maxlatitude"] - config["minlatitude"]
config["longituderange"] = config["maxlongitude"] - config["minlongitude"]

pyproj = pyimport("pyproj") # use it in an environment with pyproj
proj = pyproj.Proj(
    "+proj=sterea +lon_0=$( (config["minlongitude"] + config["maxlongitude"]) / 2 ) +lat_0=$( (config["minlatitude"] + config["maxlatitude"]) / 2 ) +units=km"
)
min_x_km, min_y_km = proj(config["minlongitude"], config["minlatitude"])
max_x_km, max_y_km = proj(config["maxlongitude"], config["maxlatitude"])

dx = ceil(abs(min_x_km)) + 2; dy = ceil(abs(min_y_km)) + 2; dz = 1; h = 1.0
m = convert(Int64, ceil((max_x_km + dx)/h)) + 3
n = convert(Int64, ceil((max_y_km + dy)/h)) + 3
l = convert(Int64, ceil((config["maxdepth"]+dz)/h)) + 1

config["xlim_km"] = [-dx, m-dx]; config["ylim_km"] = [-dy, n-dy]
config["zlim_km"] = [-dz, l-dz]; config["h"] = h

## calculate traveltime of stations and events
vel_p = ones(m,n,l) * 6.0; vel_s = ones(m,n,l) * 3.5 # (54, 61, 22)
if folder == "anomaly"
    vel_p[20:40, 20:40, 8:14] .= 6.5; vel_s[20:50, 20:40, 8:14] .= 3.8
    vel_p[5:25, 4:19, 2:7] .= 5.5; vel_s[5:25, 4:19, 2:7] .= 3.2
elseif folder == "checkerboard"
    for i = 0:m-1
        for j = 0:n-1
            for k = 0:l-1
                ii = (i-i%len)/len; jj = (j-j%len)/len; kk = (k-k%len)/len
                if (ii+jj+kk)%2 ==0
                    vel_p[i+1,j+1,k+1] = 6.5
                    vel_s[i+1,j+1,k+1] = 3.8
                else
                    vel_p[i+1,j+1,k+1] = 5.5
                    vel_s[i+1,j+1,k+1] = 3.2
                end
            end
        end
    end
end
if isfile(folder * "/velocity_p.h5")
    rm(folder * "/velocity_p.h5")
    rm(folder * "/velocity_s.h5")
end
h5write(folder * "/velocity_p.h5", "data", vel_p)
h5write(folder * "/velocity_s.h5", "data", vel_s)
config["vel_p"] = 1
config["vel_s"] = 1

open(folder * "/config.json", "w") do f
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
    latitude = config["latitude0"] + (rand() - 0.5) * config["latituderange"]
    longitude = config["longitude0"] + (rand() - 0.5) * config["longituderange"]
    elevation_m = rand() * 1000
    depth_km = -elevation_m / 1000
    push!(stations, (station_id=station_id, longitude=longitude, latitude=latitude, elevation_m=elevation_m, x_km=missing, y_km=missing, z_km=depth_km))
end

stations.x_km, stations.y_km = proj(stations.longitude, stations.latitude)

CSV.write(folder * "/stations.csv", stations)

## build events

num_event = 100
events = DataFrame(event_index=[], longitude=[], latitude=[], magnitude=[], time=[], x_km = [], y_km = [], z_km = [])
index = 0
for i in 1:num_event
    global index += 1 
    latitude = config["latitude0"] + (rand() - 0.5) * config["latituderange"]
    longitude = config["longitude0"] + (rand() - 0.5) * config["longituderange"]
    magnitude = rand()*2
    depth_km = config["depth0"] + (rand()-0.5) * 20
    time = DateTime("2019-01-01T00:00:00", "yyyy-mm-ddTHH:MM:SS")
    push!(events, (event_index=index, longitude=longitude, latitude=latitude, magnitude=magnitude, time=time, x_km=missing, y_km=missing, z_km=depth_km))
end
events.x_km, events.y_km = proj(events.longitude, events.latitude)

CSV.write(folder * "/events.csv", events)

## adjust locations for calculation

stations.x = (stations.x_km .+ dx) ./ h .+ 1; events.x = (events.x_km .+ dx) ./ h .+ 1
stations.y = (stations.y_km .+ dy) ./ h .+ 1; events.y = (events.y_km .+ dy) ./ h .+ 1
stations.z = (stations.z_km .+ dz) ./ h .+ 1; events.z = (events.z_km .+ dz) ./ h .+ 1

up = PyObject[]; us = PyObject[]; fvel_p = 1 ./ vel_p; fvel_s = 1 ./ vel_s
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
caltime_p = ones(num_station,num_event); qua_p = ones(num_station,num_event)
caltime_s = ones(num_station,num_event); qua_s = ones(num_station,num_event)
for i = 1:num_station
    for j = 1:num_event
        jx = events.x[j]; x1 = convert(Int64,floor(jx)); x2 = x1 + 1
        jy = events.y[j]; y1 = convert(Int64,floor(jy)); y2 = y1 + 1
        jz = events.z[j]; z1 = convert(Int64,floor(jz)); z2 = z1 + 1
        # P wave
        txyz = (z2-jz)*(y2-jy)*((x2-jx)*utravel_p[i][x1,y1,z1] + (jx-x1)*utravel_p[i][x2,y1,z1]) +
               (z2-jz)*(jy-y1)*((x2-jx)*utravel_p[i][x1,y2,z1] + (jx-x1)*utravel_p[i][x2,y2,z1]) + 
               (jz-z1)*(y2-jy)*((x2-jx)*utravel_p[i][x1,y1,z2] + (jx-x1)*utravel_p[i][x2,y1,z2]) + 
               (jz-z1)*(jy-y1)*((x2-jx)*utravel_p[i][x1,y2,z2] + (jx-x1)*utravel_p[i][x2,y2,z2])
        caltime_p[i,j] = txyz
        qua_p[i,j] = rand()/2 + 0.5
        # S wave
        txyz = (z2-jz)*(y2-jy)*((x2-jx)*utravel_s[i][x1,y1,z1] + (jx-x1)*utravel_s[i][x2,y1,z1]) +
               (z2-jz)*(jy-y1)*((x2-jx)*utravel_s[i][x1,y2,z1] + (jx-x1)*utravel_s[i][x2,y2,z1]) + 
               (jz-z1)*(y2-jy)*((x2-jx)*utravel_s[i][x1,y1,z2] + (jx-x1)*utravel_s[i][x2,y1,z2]) + 
               (jz-z1)*(jy-y1)*((x2-jx)*utravel_s[i][x1,y2,z2] + (jx-x1)*utravel_s[i][x2,y2,z2])
        caltime_s[i,j] = txyz
        qua_s[i,j]  = rand()/2 + 0.5
    end
end

## decide data of picks

picks = DataFrame(station_id=[], phase_time=[], phase_score=[], phase_type=[], event_index=[])
for i = 1:num_event
    for j = 1:num_station
        station_id = stations.station_id[j]
        event_index = events.event_index[i]

        type = "P"
        time = events.time[i] + Millisecond(round(1000 * caltime_p[j,i]))
        score = qua_p[j,i]
        push!(picks,(station_id=station_id, phase_time=time, phase_score=score, phase_type=type, event_index=event_index))
        type = "S"
        time = events.time[i] + Millisecond(round(1000 * caltime_s[j,i]))
        score = qua_s[j,i]
        push!(picks,(station_id=station_id, phase_time=time, phase_score=score, phase_type=type, event_index=event_index))
    end
end

CSV.write(folder * "/picks.csv", picks)


## matrix and files for inversion

folder = folder * "/inv/"; 
if !isfile(folder * "for_P/uobs_p.h5")
    mkdir(folder * "for_P"); mkdir(folder * "for_S")
end

rfile = open(folder * "range.txt","w")
println(rfile,m);println(rfile,n);println(rfile,l);println(rfile,h)
close(rfile)

CSV.write(folder * "allsta.csv", stations)
CSV.write(folder * "alleve.csv", events)

if isfile(folder * "for_P/uobs_p.h5")
    rm(folder * "for_P/uobs_p.h5"); rm(folder * "for_P/qua_p.h5")
    rm(folder * "for_S/uobs_s.h5"); rm(folder * "for_S/qua_s.h5")
    rm(folder * "uobs_d.h5")
end
h5write(folder * "for_P/uobs_p.h5","matrix",caltime_p); h5write(folder * "for_P/qua_p.h5","matrix",qua_p)
h5write(folder * "for_S/uobs_s.h5","matrix",caltime_s); h5write(folder * "for_S/qua_s.h5","matrix",qua_s)
caltime_d = caltime_s - caltime_p; h5write(folder * "uobs_d.h5","matrix",caltime_d)
