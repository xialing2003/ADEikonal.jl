using CSV
using ADCME
using DataFrames
using Serialization
using HDF5
using PyCall
using Dates
using PyPlot
using Colors
using JSON
using PyCall

region = "BayArea/"
folder = "../local/" * region * "readin_data/"
config = JSON.parsefile("../local/" * region * "readin_data/config.json")["post_rect"]

rfile = open(folder * "range.txt","r")
m = parse(Int,readline(rfile)); n = parse(Int,readline(rfile))
l = parse(Int,readline(rfile)); h = parse(Float64,readline(rfile))
dx = parse(Int,readline(rfile)); dy = parse(Int,readline(rfile)); dz = parse(Int,readline(rfile))

lambda = config["lambda_s"]
vel0 = h5read(folder * "velocity/vel0_s.h5","data")
v = h5read(folder * "velocity/vel_check_s_10.h5","data")
#folder = folder * "inv_S_"*string(lambda)
folder = folder * "check_S_10/"

sess = Session(); init(sess)
if !isdir(folder * "/plot/")
    mkdir(folder * "/plot/")
end
if !isdir(folder * "/post/")
    mkdir(folder * "/post/")
end
for ite =10:10:100
    #
    fvar = h5read(folder * "/intermediate/iter_$ite.h5","data")
    fvar = tf.reshape(fvar,(m,n,l)); fvel = run(sess,fvar)
    vel = ones(m,n,l)
    for i = 1:m
        for j = 1:n
            for k = 1:l
                vel[i,j,k] = 2*sigmoid(fvel[i,j,k])-1 + vel0[i,j,k]
            end
        end
    end
    #vel = h5read(folder * "post_$ite.h5","data")
    figure(figsize = (config["width"],config["length"]))
    for i = 1:16
        subplot(4,4,i)
        pcolormesh(transpose(vel[:,:,i]), cmap = "seismic_r",vmin=vel[1,1,i]-1,vmax=vel[1,1,i]+1)
        title("layer "*string(i))
        colorbar()
    end
    savefig(folder * "/plot/plot_$ite.png")
    close()
    if !isfile(folder * "/post/post_$ite.h5")
        h5write(folder * "/post/post_$ite.h5","data",vel)
    end
end

folder = "/home/lingxia/ADTomo.jl/local/BayArea/readin_data/check_S_10/output_tit"
width = 4; length = 8
for i = 1:16
    figure(figsize=(width,length))
    pcolormesh(transpose(v[:,:,i]), cmap = "seismic_r", vmin = v[1,1,i]-1,vmax = v[1,1,i]+1)
    title(string((i-2)*2)*" km",fontsize = "xx-large")
    colorbar()
    plt.tight_layout()
    savefig(folder * "/layer_$i.png")
end

using HDF5
using PyPlot
function post_plot(v,pvs,num)
    figure(figsize=(20,30))
    for i = 1:16
        subplot(4,4,i)
        pcolormesh(transpose(v[:,:,i]),cmap="seismic_r",vmax=v[1,1,i]+1,vmin=v[1,1,i]-1)
        title("layer " * string(i))
        colorbar()
    end
    tight_layout()
    savefig("plot/Vp_"*num*".png")
    close()

    figure(figsize=(20,30))
    for i = 1:16
        subplot(4,4,i)
        pcolormesh(transpose(pvs[:,:,i]),cmap="seismic_r",vmax=2.1,vmin=1.4)
        title("layer " * string(i))
        colorbar()
    end
    tight_layout()
    savefig("plot/pvs_"*num*".png")
    close()
end

for i = 20:20:280
    v = h5read("vp/Vp_$i.h5","data")
    pvs = h5read("pvs/pvs_$i.h5","data")
    post_plot(v,pvs,string(i))
end


using HDF5
using ADCME
vel0 = h5read("/home/lingxia/ADTomo.jl/local/BayArea/readin_data/velocity/vel0_p.h5","data")
function post(f,num)
    vp_ = f[:,:,1:17]; pvs_ = f[:,:,18:34]
    vp = zero(vp_); pvs = zero(pvs_)
    (m,n,l) = size(vp)
    for i = 1:m
        for j = 1:n
            for k = 1:l
                vp[i,j,k] = 2*sigmoid(vp_[i,j,k])-1 + vel0[i,j,k]
                pvs[i,j,k] = 4*sigmoid(pvs_[i,j,k])-2 + 1.7583
            end
        end
    end
    h5write("vp/Vp_"*num*".h5","data",vp)
    h5write("pvs/pvs_"*num*".h5","data",pvs)
end

sess = Session(); init(sess)
for i = 20:20:280
    f_ = h5read("intermediate/iter_$i.h5","data")
    f_ = tf.reshape(f_,(82,190,34))
    f = run(sess,f_)
    post(f,string(i))
end

f = h5read("final.h5","data")
post(f,"final")