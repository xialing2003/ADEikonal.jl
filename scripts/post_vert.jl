using HDF5
using PyPlot
using PyCall
np = pyimport("numpy") 

x = np.arange(0, 17, 2)
y1 = np.arange(1, 191, 1)
y2 = np.arange(1, 83, 1)
folder = "../local/BayArea/readin_data/"
xrange = [23, 36, 45, 58, 71]; yrange = [51, 75, 95, 108, 127, 138]
dicx = Dict(); dicx[71]="A"; dicx[58]="B";dicx[45]="C";dicx[36]="D";dicx[23]="E"
dicy = Dict(); dicy[51]="A"; dicy[75]="B";dicy[95]="C";dicy[108]="D";dicy[127]="E";dicy[138]="F"

if !isdir(folder*"inv_P_0.03/vertical/all_x")
    mkdir(folder*"inv_P_0.03/vertical/")
    mkdir(folder*"inv_P_0.03/vertical/all_x")
    mkdir(folder*"inv_P_0.03/vertical/all_y")
end
if !isdir(folder*"inv_S_0.1/vertical/all_x")
    mkdir(folder*"inv_S_0.1/vertical/")
    mkdir(folder*"inv_S_0.1/vertical/all_x")
    mkdir(folder*"inv_S_0.1/vertical/all_y")
end

v = h5read(folder * "inv_P_0.03/post/post_100.h5","data")
for i in xrange
    local fig = figure(figsize=(15,2))
    local ax = fig.add_subplot(111)
    local pcm = ax.pcolormesh(y1, x, transpose(v[i,:,2:10]), cmap="turbo_r")
    subplots_adjust(left=0.04, right=0.95, top=0.8, bottom=0.2)
    local cax = fig[:add_axes]([0.96, 0.1, 0.02, 0.8])
    colorbar(pcm, cax = cax)
    ax.invert_yaxis()
    ax.set_ylabel("depth/km")
    ax.text(-5, -2, dicx[i], fontsize=20, color="black")
    savefig(folder * "inv_P_0.03/vertical/all_x/$i.png")
    close()
end
for i in yrange
    local fig = figure(figsize=(10,2))
    local ax = fig.add_subplot(111)
    local pcm = ax.pcolormesh(y2, x, transpose(v[:,i,2:10]), cmap="turbo_r")
    subplots_adjust(left=0.05, right=0.93, top=0.8, bottom=0.2)
    local cax = fig[:add_axes]([0.94, 0.1, 0.02, 0.8])
    colorbar(pcm, cax = cax)
    ax.invert_yaxis()
    ax.set_ylabel("depth/km")
    ax.text(-2, -2, dicy[i], fontsize=20, color="black")
    savefig(folder * "inv_P_0.03/vertical/all_y/$i.png")
    close()
end

v = h5read(folder * "inv_S_0.1/post/post_100.h5","data")
for i in xrange
    local fig = figure(figsize=(15,2))
    local ax = fig[:add_subplot](111)
    local pcm = ax[:pcolormesh](y1,x,transpose(v[i,:,2:10]),cmap="turbo_r")
    subplots_adjust(left=0.04, right=0.95, top=0.8, bottom=0.2)
    local cax = fig[:add_axes]([0.96, 0.1, 0.02, 0.8])
    colorbar(pcm, cax = cax)
    ax.invert_yaxis()
    ax.set_ylabel("depth/km")
    ax.text(-5, -2, dicx[i], fontsize=20, color="black")
    savefig(folder * "inv_S_0.1/vertical/all_x/$i.png")
    close()
end
for i in yrange
    local fig = figure(figsize=(10,2))
    local ax = fig[:add_subplot](111)
    local pcm = ax[:pcolormesh](y2,x,transpose(v[:,i,2:10]),cmap="turbo_r")
    subplots_adjust(left=0.05, right=0.93, top=0.8, bottom=0.2)
    local cax = fig[:add_axes]([0.94, 0.1, 0.02, 0.8])
    colorbar(pcm, cax = cax)
    ax.invert_yaxis()
    ax.set_ylabel("depth/km")
    ax.text(-2, -2, dicy[i], fontsize=20, color="black")
    savefig(folder * "inv_S_0.1/vertical/all_y/$i.png")
    close()
end
