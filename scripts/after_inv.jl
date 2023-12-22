push!(LOAD_PATH,"../src")
using CSV
using DataFrames
using HDF5
using ADCME
using ADTomo
using PyCall
using Dates
using PyPlot
using JSON
using Random
using Optim
using LineSearches
Random.seed!(233)

region = "BayArea/"
folder = "../local/" * region * "readin_data/"
config = JSON.parsefile(folder * "config.json")["inversion"]
rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))

allsta = CSV.read(folder * "sta_eve/allsta.csv",DataFrame); numsta = size(allsta,1)
alleve = CSV.read(folder * "sta_eve/alleve.csv",DataFrame); numeve = size(alleve,1)
uobs_p = h5read(folder * "for_P/uobs_p.h5","matrix")
qua_p = h5read(folder * "for_P/qua_p.h5","matrix")
uobs_s = h5read(folder * "for_S/uobs_s.h5","matrix")
qua_s = h5read(folder * "for_S/qua_s.h5","matrix")

folder = folder * "joint_1.75_0.03_0.01/sig_1_2/"
vp = h5read(folder * "Vp.h5","data")
pvs = h5read(folder * "pvs.h5","data")
uvar_p = PyObject[]; uvar_s = PyObject[]
for i = 1:numsta
    ix = allsta.x[i]; ixu = convert(Int64,ceil(ix)); ixd = convert(Int64,floor(ix))
    iy = allsta.y[i]; iyu = convert(Int64,ceil(iy)); iyd = convert(Int64,floor(iy))
    iz = allsta.z[i]; izu = convert(Int64,ceil(iz)); izd = convert(Int64,floor(iz))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*h/vp[ixu,iyu,izu]
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*h/vp[ixu,iyu,izd]
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*h/vp[ixu,iyd,izu]
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*h/vp[ixu,iyd,izd]
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*h/vp[ixd,iyu,izu]
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*h/vp[ixd,iyu,izd]
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*h/vp[ixd,iyd,izu]
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*h/vp[ixd,iyd,izd]
    push!(uvar_p,eikonal3d(u0,1 ./ vp,h,m,n,l,1e-3,false))

    u0 = 1000 * ones(m,n,l)
    u0[ixu,iyu,izu] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izu)^2)*h/vp[ixu,iyu,izu]*pvs[ixu,iyu,izu]
    u0[ixu,iyu,izd] = sqrt((ix-ixu)^2+(iy-iyu)^2+(iz-izd)^2)*h/vp[ixu,iyu,izd]*pvs[ixu,iyu,izd]
    u0[ixu,iyd,izu] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izu)^2)*h/vp[ixu,iyd,izu]*pvs[ixu,iyd,izu]
    u0[ixu,iyd,izd] = sqrt((ix-ixu)^2+(iy-iyd)^2+(iz-izd)^2)*h/vp[ixu,iyd,izd]*pvs[ixu,iyd,izd]
    u0[ixd,iyu,izu] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izu)^2)*h/vp[ixd,iyu,izu]*pvs[ixd,iyu,izu]
    u0[ixd,iyu,izd] = sqrt((ix-ixd)^2+(iy-iyu)^2+(iz-izd)^2)*h/vp[ixd,iyu,izd]*pvs[ixd,iyu,izd]
    u0[ixd,iyd,izu] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izu)^2)*h/vp[ixd,iyd,izu]*pvs[ixd,iyd,izu]
    u0[ixd,iyd,izd] = sqrt((ix-ixd)^2+(iy-iyd)^2+(iz-izd)^2)*h/vp[ixd,iyd,izd]*pvs[ixd,iyd,izd]
    push!(uvar_s,eikonal3d(u0,pvs ./ vp,h,m,n,l,1e-3,false))


end

sess = Session(); init(sess)
ucal_p = run(sess,uvar_p); caltime_p = ones(numsta,numeve)
ucal_s = run(sess,uvar_s); caltime_s = ones(numsta,numeve)
for i = 1:numsta
    for j = 1:numeve
        jx = alleve.x[j]; x1 = convert(Int64,floor(jx)); x2 = convert(Int64,ceil(jx))
        jy = alleve.y[j]; y1 = convert(Int64,floor(jy)); y2 = convert(Int64,ceil(jy))
        jz = alleve.z[j]; z1 = convert(Int64,floor(jz)); z2 = convert(Int64,ceil(jz))
        # P wave
        if x1 == x2
            tx11 = ucal_p[i][x1,y1,z1]; tx12 = ucal_p[i][x1,y1,z2]
            tx21 = ucal_p[i][x1,y2,z1]; tx22 = ucal_p[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*ucal_p[i][x1,y1,z1] + (jx-x1)*ucal_p[i][x2,y1,z1]
            tx12 = (x2-jx)*ucal_p[i][x1,y1,z2] + (jx-x1)*ucal_p[i][x2,y1,z2]
            tx21 = (x2-jx)*ucal_p[i][x1,y2,z1] + (jx-x1)*ucal_p[i][x2,y2,z1]
            tx22 = (x2-jx)*ucal_p[i][x1,y2,z2] + (jx-x1)*ucal_p[i][x2,y2,z2]
        end
        if y1 == y2
            txy1 = tx11; txy2 = tx12
        else
            txy1 = (y2-jy)*tx11 + (jy-y1)*tx21
            txy2 = (y2-jy)*tx12 + (jy-y1)*tx22
        end
        if z1 ==z2
            txyz = txy1
        else
            txyz = (z2-jz)*txy1 + (jz-z1)*txy2
        end
        caltime_p[i,j] = txyz
        # S wave
        if x1 == x2
            tx11 = ucal_s[i][x1,y1,z1]; tx12 = ucal_s[i][x1,y1,z2]
            tx21 = ucal_s[i][x1,y2,z1]; tx22 = ucal_s[i][x1,y2,z2]
        else
            tx11 = (x2-jx)*ucal_s[i][x1,y1,z1] + (jx-x1)*ucal_s[i][x2,y1,z1]
            tx12 = (x2-jx)*ucal_s[i][x1,y1,z2] + (jx-x1)*ucal_s[i][x2,y1,z2]
            tx21 = (x2-jx)*ucal_s[i][x1,y2,z1] + (jx-x1)*ucal_s[i][x2,y2,z1]
            tx22 = (x2-jx)*ucal_s[i][x1,y2,z2] + (jx-x1)*ucal_s[i][x2,y2,z2]
        end
        if y1 == y2
            txy1 = tx11; txy2 = tx12
        else
            txy1 = (y2-jy)*tx11 + (jy-y1)*tx21
            txy2 = (y2-jy)*tx12 + (jy-y1)*tx22
        end
        if z1 ==z2
            txyz = txy1
        else
            txyz = (z2-jz)*txy1 + (jz-z1)*txy2
        end
        caltime_s[i,j] = txyz
    end
end

# residual P
sum_loss_p = PyObject[]
for i = 1:numsta
    for j = 1:numeve
        if uobs_p[i,j] != -1
            push!(sum_loss_p, qua_p[i,j]*(uobs_p[i,j]-caltime_p[i,j])^2)
        end
    end
end
@show loss_res_p = sum(sum_loss_p)

# residual S
sum_loss_s = PyObject[]
for i = 1:numsta
    for j = 1:numeve
        if uobs_s[i,j] != -1
            push!(sum_loss_s, qua_s[i,j]*(uobs_s[i,j]-caltime_s[i,j])^2)
        end
    end
end
@show loss_res_s = sum(sum_loss_s)

sh1 = 5; sh2 = convert(Int,(sh1-1)/2)
sv1 = 3; sv2 = convert(Int,(sv1-1)/2)
gauss_wei = ones(sh1,sh1,sv1) ./ (sh1*sh1*sv1)
filter = tf.constant(gauss_wei,shape=(sh1,sh1,sv1,1,1),dtype=tf.float64)

# regularization P
var = tf.reshape(vp,(m,n,l))
var_ = tf.reshape(vp,(1,m,n,l,1))
cvar = tf.nn.conv3d(var_, filter, strides=(1,1,1,1,1), padding="VALID")
n_var = tf.reshape(cvar, (m-sh1+1,n-sh1+1,l-sv1+1))

reg_p = sum(abs(var[sh2+1:m-sh2,sh2+1:n-sh2,sv2+1:l-sv2]-n_var))
@show run(sess, reg_p)
# regularizaiton S
vs = vp ./ pvs
var = tf.reshape(vs,(m,n,l))
var_ = tf.reshape(vs,(1,m,n,l,1))
cvar = tf.nn.conv3d(var_, filter, strides=(1,1,1,1,1), padding="VALID")
n_var = tf.reshape(cvar, (m-sh1+1,n-sh1+1,l-sv1+1))

reg_s = sum(abs(var[sh2+1:m-sh2,sh2+1:n-sh2,sv2+1:l-sv2]-n_var))
@show run(sess, reg_s)
# regularization pvs
var = tf.reshape(pvs,(m,n,l))
var_ = tf.reshape(pvs,(1,m,n,l,1))
cvar = tf.nn.conv3d(var_, filter, strides=(1,1,1,1,1), padding="VALID")
n_var = tf.reshape(cvar, (m-sh1+1,n-sh1+1,l-sv1+1))

reg_pvs = sum(abs(var[sh2+1:m-sh2,sh2+1:n-sh2,sv2+1:l-sv2]-n_var))
@show run(sess, reg_pvs)
