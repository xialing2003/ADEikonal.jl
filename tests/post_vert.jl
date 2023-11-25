using HDF5
using PyPlot
using PyCall
np = pyimport("numpy") 

v = h5read("readin_data/store/new4/2/inv_S_0.1/intermediate/post_101.h5","data")
x = np.arange(0, 17, 2)
y1 = np.arange(1, 191, 1)
y2 = np.arange(1, 83, 1)
#=
i = 1
fig = figure(figsize=(10, 2))
ax = fig.add_subplot(111)
pcm = ax.pcolormesh(y, x, transpose(v[i,:,2:10]), cmap="turbo_r")
colorbar(pcm)
ax.invert_yaxis()
savefig("readin_data/store/new4/2/vertical/S_all_x/$i.png")
close()
=#


for i = 1:82
    local fig = figure(figsize=(15,2))
    local ax = fig.add_subplot(111)
    local pcm = ax.pcolormesh(y1, x, transpose(v[i,:,2:10]), cmap="turbo_r")
    colorbar(pcm)
    ax.invert_yaxis()
    savefig("readin_data/store/new4/2/vertical/S_all_x/$i.png")
    close()
end

for i = 1:190
    local fig = figure(figsize=(10,2))
    local ax = fig.add_subplot(111)
    local pcm = ax.pcolormesh(y2, x, transpose(v[:,i,2:10]), cmap="turbo_r")
    colorbar(pcm)
    ax.invert_yaxis()
    savefig("readin_data/store/new4/2/vertical/S_all_y/$i.png")
    close()
end

v = h5read("readin_data/store/new4/2/inv_P_0.03/intermediate/post_101.h5","data")
for i = 1:82
    local fig = figure(figsize=(15,2))
    local ax = fig[:add_subplot](111)
    local pcm = ax[:pcolormesh](y1,x,transpose(v[i,:,2:10]),cmap="turbo_r")
    colorbar(pcm)
    ax.invert_yaxis()
    savefig("readin_data/store/new4/2/vertical/P_all_x/$i.png")
    close()
end
for i = 1:190
    local fig = figure(figsize=(10,2))
    local ax = fig[:add_subplot](111)
    local pcm = ax[:pcolormesh](y2,x,transpose(v[:,i,2:10]),cmap="turbo_r")
    colorbar(pcm)
    ax.invert_yaxis()
    savefig("readin_data/store/new4/2/vertical/P_all_y/$i.png")
    close()
end
