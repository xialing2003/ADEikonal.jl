push!(LOAD_PATH,"../src")
ENV["TF_NUM_INTEROP_THREADS"] = 1
using ADCME
using ADTomo
using PyCall
using PyPlot
using CSV
using DataFrames
using HDF5
using LinearAlgebra
using JSON
using Random
using Optim
using LineSearches
Random.seed!(233)

mpi_init()
rank = mpi_rank()
nproc = mpi_size()

region = "BayArea/"
folder = "../local/" * region * "readin_data/"
config = JSON.parsefile("../local/" * region * "readin_data/config.json")["inversion"]

rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile))
dz = parse(Int,readline(rfile)); pvs_old = 1.7583

allsta = CSV.read(folder * "sta_eve/allsta.csv",DataFrame); numsta = size(allsta,1)
alleve = CSV.read(folder * "sta_eve/alleve.csv",DataFrame); numeve = size(alleve,1)
vel0 = h5read(folder * "velocity/vel0_p.h5","data")

uobs_p = h5read(folder * "for_P/uobs_p.h5","matrix")
qua_p = h5read(folder * "for_P/qua_p.h5","matrix")
uobs_d = h5read(folder * "uobs_d.h5","matrix")
qua_d = h5read(folder * "qua_d.h5","matrix")

allsta = allsta[rank+1:nproc:numsta,:]
uobs_p = uobs_p[rank+1:nproc:numsta,:]
qua_p = qua_p[rank+1:nproc:numsta,:]
uobs_d = uobs_d[rank+1:nproc:numsta,:]
qua_d = qua_d[rank+1:nproc:numsta,:]
numsta = size(allsta,1)

vari = mpi_bcast(Variable(zeros(m,n,l*2)))
fvar = 2*sigmoid(vari[:,:,1:l])-1 + vel0
pvs = 0.6*sigmoid(vari[:,:,1+l:2*l])-0.3 + ones(Float64,m,n,l)*pvs_old

uvar_p = PyObject[]; uvar_s = PyObject[]
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*h/vel0[ixu,iyu,izu]
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*h/vel0[ixu,iyu,izd]
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*h/vel0[ixu,iyd,izu]
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*h/vel0[ixu,iyd,izd]
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*h/vel0[ixd,iyu,izu]
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*h/vel0[ixd,iyu,izd]
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*h/vel0[ixd,iyd,izu]
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*h/vel0[ixd,iyd,izd]
    push!(uvar_p,eikonal3d(u0,1 ./ fvar,h,m,n,l,1e-3,false))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*h/vel0[ixu,iyu,izu]*pvs_old
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*h/vel0[ixu,iyu,izd]*pvs_old
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*h/vel0[ixu,iyd,izu]*pvs_old
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*h/vel0[ixu,iyd,izd]*pvs_old
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*h/vel0[ixd,iyu,izu]*pvs_old
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*h/vel0[ixd,iyu,izd]*pvs_old
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*h/vel0[ixd,iyd,izu]*pvs_old
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*h/vel0[ixd,iyd,izd]*pvs_old
    push!(uvar_s,eikonal3d(u0,pvs ./ fvar,h,m,n,l,1e-3,false))
end

caltime_p = []; caltime_s = []
for i = 1:numsta
    timei_p = []; timei_s = []
    for j = 1:numeve

        if uobs_p[i,j] == -1
            push!(timei_p,Variable(-1))
            push!(timei_s,Variable(-1))
            continue
        end 

        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = x1 + 1
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = y1 + 1
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = z1 + 1

        txyz = (z2-jz)*(y2-jy)*((x2-jx)*uvar_p[i][x1,y1,z1] + (jx-x1)*uvar_p[i][x2,y1,z1]) +
               (z2-jz)*(jy-y1)*((x2-jx)*uvar_p[i][x1,y1,z2] + (jx-x1)*uvar_p[i][x2,y1,z2]) + 
               (jz-z1)*(y2-jy)*((x2-jx)*uvar_p[i][x1,y2,z1] + (jx-x1)*uvar_p[i][x2,y2,z1]) + 
               (jz-z1)*(jy-y1)*((x2-jx)*uvar_p[i][x1,y2,z2] + (jx-x1)*uvar_p[i][x2,y2,z2])
        push!(timei_p,txyz)

        if uobs_d[i,j] == -1
            push!(timei_s,Variable(-1))
            continue
        end

        txyz = (z2-jz)*(y2-jy)*((x2-jx)*uvar_s[i][x1,y1,z1] + (jx-x1)*uvar_s[i][x2,y1,z1]) +
               (z2-jz)*(jy-y1)*((x2-jx)*uvar_s[i][x1,y1,z2] + (jx-x1)*uvar_s[i][x2,y1,z2]) + 
               (jz-z1)*(y2-jy)*((x2-jx)*uvar_s[i][x1,y2,z1] + (jx-x1)*uvar_s[i][x2,y2,z1]) + 
               (jz-z1)*(jy-y1)*((x2-jx)*uvar_s[i][x1,y2,z2] + (jx-x1)*uvar_s[i][x2,y2,z2])
        push!(timei_s,txyz)
    end
    push!(caltime_p,timei_p)
    push!(caltime_s,timei_s)
end

sum_loss_time = PyObject[];
for i = 1:numeve
    for j = 1:numsta
        if uobs_p[j,i] != -1
            push!(sum_loss_time, qua_p[j,i]*(uobs_p[j,i]-caltime_p[j][i])^2)
        end
        if uobs_d[j,i] != -1
            push!(sum_loss_time, qua_d[j,i]*(uobs_d[j,i]-caltime_s[j][i]+caltime_p[j][i])^2)
        end
    end
end
#
sh1 = config["smooth_hor"]; sv1 = config["smooth_ver"]
sh2 = convert(Int,(sh1-1)/2); sv2 = convert(Int,(sv1-1)/2)
gauss_wei = ones(sh1,sh1,sv1) ./ (sh1*sh1*sv1)
filter = tf.constant(gauss_wei,shape=(sh1,sh1,sv1,1,1),dtype=tf.float64)

o_vp = tf.reshape(fvar, (1,m,n,l,1))
cvp = tf.nn.conv3d(o_vp, filter, strides = (1,1,1,1,1), padding="VALID")
n_vp = tf.reshape(cvp, (m-sh1+1,n-sh1+1,l-sv1+1))

vs_ =  fvar ./ pvs; o_vs = tf.reshape(vs_, (1,m,n,l,1));
cvs = tf.nn.conv3d(o_vs, filter, strides = (1,1,1,1,1), padding="VALID")
n_vs = tf.reshape(cvs, (m-sh1+1,n-sh1+1,l-sv1+1))

o_pvs = tf.reshape(pvs, (1,m,n,l,1))
cpvs = tf.nn.conv3d(o_pvs, filter, strides = (1,1,1,1,1), padding="VALID")
n_pvs = tf.reshape(cpvs, (m-sh1+1,n-sh1+1,l-sv1+1))
#

sess = Session(); init(sess)
#loss = sum(sum_loss_time) + 0.01 * sum((fvar - n_vp)^4) + 0.01 * sum((pvs - n_pvs)^4) #+ 0.001 * sum(abs(vs_ - n_vs)) 
loss = sum(sum_loss_time) + 
    0.002 * sum(abs(fvar[sh2+1:m-sh2,sh2+1:n-sh2,sv2+1:l-sv2] - n_vp)) +
    0.006 * sum(abs(pvs[sh2+1:m-sh2,sh2+1:n-sh2,sv2+1:l-sv2] - n_pvs)) + 
    0.002 * sum(abs(vs_[sh2+1:m-sh2,sh2+1:n-sh2,sv2+1:l-sv2] - n_vs)) +
    0.0005 * sum(abs(4*sigmoid(vari[:,:,1+l:2*l])-2))
loss = mpi_sum(loss)

options = Optim.Options(iterations = 1000)
# loc = folder * "test_order/p40.01_pvs40.01/"
loc = folder * "reg_1_0.5/0.002_0.006_0.002_0.0005/"
result = ADTomo.mpi_optimize(sess, loss, method="LBFGS", options = options, 
    loc = loc*"intermediate/", steps = 20)
if mpi_rank()==0
    @info [size(result[i]) for i = 1:length(result)]
    @info [length(result)]
    h5write(loc * "final.h5","data",result[1])
end
mpi_finalize()