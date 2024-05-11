using CSV
using DataFrames
using HDF5
using PyCall
using ADCME
using ADTomo
using Random

region = "synthetic/"
folder = "../local/" * region * "readin_data/"
if !isdir(folder) mkdir(folder) end 

m = 20; n = 40; l = 10; h = 1.0
dx = 0; dy = 0; dz = 0
rfile = open(folder * "range.txt","w")
println(rfile,m);println(rfile,n);println(rfile,l);println(rfile,h)
println(rfile,dx);println(rfile,dy);println(rfile,dz)
close(rfile)

folder_ = folder * "sta_eve/"
if !isdir(folder_) mkdir(folder_) end 

numsta = 100; allsta = DataFrame(x=[], y=[], z=[])
numeve = 200; alleve = DataFrame(x=[], y=[], z=[])
for i = 1:numsta
    push!(allsta.x, rand()*(m-1)+1)
    push!(allsta.y, rand()*(n-1)+1)
    push!(allsta.z, rand()+1)
end
for i = 1:numeve
    push!(alleve.x, rand()*(m-1)+1)
    push!(alleve.y, rand()*(n-1)+1)
    push!(alleve.z, rand()*(l-1)+1)
end
CSV.write(folder_ * "alleve.csv",alleve)
CSV.write(folder_ * "allsta.csv",allsta)

folder_ = folder * "velocity/"
if !isdir(folder_) mkdir(folder_) end

# ### 
# Vp & Vs is designed because I suppose the Vp/Vs ratio equals 1.7583 except in anomalies
# ###
Vp = ones(m,n,l); Vs = ones(m,n,l)
Vp[:,:,1] .= 2.63745; Vs[:,:,1] .= 1.5
Vp[:,:,2:3] .= 4.21992; Vs[:,:,2:3] .= 2.4
Vp[:,:,4:6] .= 5.62656; Vs[:,:,4:6] .= 3.2
Vp[:,:,7:10] .= 6.85737; Vs[:,:,7:10] .= 3.9

if isfile(folder_*"vel0_p.h5")
    rm(folder_*"vel0_p.h5")
    rm(folder_*"vel0_s.h5")
    rm(folder_*"velp_true.h5")
    rm(folder_*"vels_true.h5")
end
h5write(folder_*"vel0_p.h5","data",Vp)
h5write(folder_*"vel0_s.h5","data",Vs)

Vp[12:16,32:40,3:6] .= 3.8; Vs[12:16,32:40,3:6] .= 2.0
Vp[2:6,5:10,1:2] .= 5.2; Vs[2:6,5:10,1:2] .= 3.1
Vp[5:10,20:30,7:9] .= 7.2; Vs[5:10,20:30,7:9] .= 4.2

h5write(folder_*"velp_true.h5","data",Vp)
h5write(folder_*"vels_true.h5","data",Vs)
if !isdir(folder*"/for_P/") 
    mkdir(folder*"/for_P/")
    mkdir(folder*"/for_S/")
end

u_p = PyObject[]; u_s = PyObject[]; fvel_p = 1 ./ Vp; fvel_s = 1 ./ Vs
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))
    slowness_pu = fvel_p[ixu,iyu,izu]; slowness_pd = fvel_p[ixu,iyu,izd]
    slowness_su = fvel_s[ixu,iyu,izu]; slowness_sd = fvel_s[ixu,iyu,izd]
    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_pd*h
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_pd*h
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_pd*h
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_pu*h
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_pd*h
    push!(u_p,eikonal3d(u0,fvel_p,h,m,n,l,1e-3,false))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_su*h
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_sd*h
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_su*h
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_sd*h
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*slowness_su*h
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*slowness_sd*h
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*slowness_su*h
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*slowness_sd*h
    push!(u_s,eikonal3d(u0,fvel_s,h,m,n,l,1e-3,false))
end

sess = Session(); init(sess)
ubeg_p = run(sess,u_p); ubeg_s = run(sess,u_s)
scaltime_p = -ones(numsta,numeve); qua_p = ones(numsta,numeve)
scaltime_s = -ones(numsta,numeve); qua_s = ones(numsta,numeve)
for i = 1:numsta
    for j = 1:numeve
        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = convert(Int64,ceil(jx))
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = convert(Int64,ceil(jy))
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = convert(Int64,ceil(jz))
        # P wave
        txyz = (z2-jz)*(y2-jy)*((x2-jx)*ubeg_p[i][x1,y1,z1] + (jx-x1)*ubeg_p[i][x2,y1,z1]) +
               (z2-jz)*(jy-y1)*((x2-jx)*ubeg_p[i][x1,y2,z1] + (jx-x1)*ubeg_p[i][x2,y2,z1]) + 
               (jz-z1)*(y2-jy)*((x2-jx)*ubeg_p[i][x1,y1,z2] + (jx-x1)*ubeg_p[i][x2,y1,z2]) + 
               (jz-z1)*(jy-y1)*((x2-jx)*ubeg_p[i][x1,y2,z2] + (jx-x1)*ubeg_p[i][x2,y2,z2])
        scaltime_p[i,j] = txyz
        # S wave
        txyz = (z2-jz)*(y2-jy)*((x2-jx)*ubeg_s[i][x1,y1,z1] + (jx-x1)*ubeg_s[i][x2,y1,z1]) +
               (z2-jz)*(jy-y1)*((x2-jx)*ubeg_s[i][x1,y2,z1] + (jx-x1)*ubeg_s[i][x2,y2,z1]) + 
               (jz-z1)*(y2-jy)*((x2-jx)*ubeg_s[i][x1,y1,z2] + (jx-x1)*ubeg_s[i][x2,y1,z2]) + 
               (jz-z1)*(jy-y1)*((x2-jx)*ubeg_s[i][x1,y2,z2] + (jx-x1)*ubeg_s[i][x2,y2,z2])
        scaltime_s[i,j] = txyz
    end
end

scaltime_d = scaltime_s - scaltime_p

if isfile(folder * "for_P/uobs_p.h5")
    rm(folder * "for_P/uobs_p.h5")
    rm(folder * "for_P/qua_p.h5")
    rm(folder * "for_S/uobs_s.h5")
    rm(folder * "for_S/qua_s.h5")
    rm(folder * "uobs_d.h5")
end
h5write(folder * "for_P/uobs_p.h5","matrix",scaltime_p)
h5write(folder * "for_S/uobs_s.h5","matrix",scaltime_s)
h5write(folder * "for_P/qua_p.h5","matrix",qua_p)
h5write(folder * "for_S/qua_s.h5","matrix",qua_s)
h5write(folder * "uobs_d.h5","matrix",scaltime_d)

# during inversion, vp can change in [-1,1], pvs can change in [-0.3,0.3]
# I suppose that Vp/Vs ratio almost equals an constant, and varies in some regions
# so Vp/Vs starts with this constant value, and I add a regression in the loss function
# just like L1 norm
# 


# ### !!!!!!!!!
# use x2 = x1 + 1 instead
# u0, f use f[ixu,iyu,izu]