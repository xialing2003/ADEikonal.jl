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

# region = "BayArea/"
# folder = "../local/" * region * "readin_data/"
folder = "../examples/anomaly/inv/"
# config = JSON.parsefile("../local/" * region * "readin_data/config.json")["inversion"]

rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))

allsta = CSV.read(folder * "sta_eve/allsta.csv",DataFrame); numsta = size(allsta,1)
alleve = CSV.read(folder * "sta_eve/alleve.csv",DataFrame); numeve = size(alleve,1)
vel0 = h5read(folder * "velocity/vel0_p.h5","data")
# vel0 = ones(m,n,l)*6.0
uobs = h5read(folder * "for_P/uobs_p.h5","matrix")
qua = h5read(folder * "for_P/qua_p.h5","matrix")

allsta = allsta[rank+1:nproc:numsta,:]
uobs = uobs[rank+1:nproc:numsta,:]
qua = qua[rank+1:nproc:numsta,:]
numsta = size(allsta,1)
#@show rank, nproc, numsta

var_change = Variable(zero(vel0))
fvar_ = sigmoid(var_change)-0.5 + vel0
fvar = mpi_bcast(fvar_)

uvar = PyObject[]
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
    push!(uvar,eikonal3d(u0,1 ./ fvar,h,m,n,l,1e-3,false))
end

caltime = []
for i = 1:numsta
    timei = []
    for j = 1:numeve
        if uobs[i,j] == -1
            push!(timei, Variable(-1))
            continue
        end
        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = x1 + 1
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = y1 + 1
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = z1 + 1

        txyz = (z2-jz)*(y2-jy)*((x2-jx)*uvar[i][x1,y1,z1] + (jx-x1)*uvar[i][x2,y1,z1]) +
               (z2-jz)*(jy-y1)*((x2-jx)*uvar[i][x1,y2,z1] + (jx-x1)*uvar[i][x2,y2,z1]) + 
               (jz-z1)*(y2-jy)*((x2-jx)*uvar[i][x1,y1,z2] + (jx-x1)*uvar[i][x2,y1,z2]) + 
               (jz-z1)*(jy-y1)*((x2-jx)*uvar[i][x1,y2,z2] + (jx-x1)*uvar[i][x2,y2,z2])
        push!(timei,txyz)
    end
    push!(caltime,timei)
end

sum_loss_time = PyObject[]
for i = 1:numeve
    for j = 1:numsta
        if uobs[j,i] == -1
            continue
        end
        push!(sum_loss_time, qua[j,i]*(uobs[j,i]-caltime[j][i])^2)
    end
end

sh1 = config["smooth_hor"]; sh2 = convert(Int,(config["smooth_hor"]-1)/2)
sv1 = config["smooth_ver"]; sv2 = convert(Int,(config["smooth_ver"]-1)/2);
gauss_wei = ones(sh1,sh1,sv1) ./ (sh1*sh1*sv1)
# gauss_wei = h5read("../local/BayArea/readin_data/filter/center.h5","data")
# (sh1,sh1,sv1) = size(gauss_wei)
# sh2 = convert(Int,(sh1-1)/2); sv2 = convert(Int,(sv1-1)/2)
filter = tf.constant(gauss_wei,shape=(sh1,sh1,sv1,1,1),dtype=tf.float64)

o_vel = fvar
o_vel = tf.concat([o_vel[m-sh2+1:m,:,:],o_vel,o_vel[1:sh2,:,:]],axis=0)
o_vel = tf.concat([o_vel[:,n-sh2+1:n,:],o_vel,o_vel[:,1:sh2,:]],axis=1)
o_vel = tf.concat([o_vel[:,:,l-sv2+1:l],o_vel,o_vel[:,:,1:sv2]],axis=2)
vel = tf.reshape(o_vel,(1,m+sh1-1,n+sh1-1,l+sv1-1,1))

cvel = tf.nn.conv3d(vel,filter,strides = (1,1,1,1,1),padding="VALID")
n_vel = tf.reshape(cvel,(m,n,l))

sess = Session(); init(sess)
loss = sum(sum_loss_time) #+ 0.01*sum(abs(fvar - n_vel))
loss = mpi_sum(loss)

options = Optim.Options(iterations = 100)
#loc = folder * "check_P_"*string(config["lambda_p"])*"/"
loc = folder * "inv_P/"
result = ADTomo.mpi_optimize(sess, loss, method="LBFGS", options = options, 
    loc = loc*"intermediate/", steps = 10)
answer = ones(m,n,l)
if mpi_rank()==0
    @info [size(result[i]) for i = 1:length(result)]
    h5write(loc * "result1.h5","data", result[1])
    for i = 1:m
        for j = 1:n
            for k = 1:l
                answer[i,j,k] = 2*sigmoid(result[1][i,j,k])-1 + vel0[i,j,k]
            end
        end
    end
    h5write(loc * "Vp.h5","data",answer)
end
mpi_finalize()