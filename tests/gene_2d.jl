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
reset_default_graph()

mpi_init()
rank = mpi_rank()
nproc = mpi_size()

m = 40           #width
n = 30           #length
h = 1.0          #resolution

f1 = ones(n,m)
f1[3:7, 5:10] .= 2
f1[14:18, 20:26] .= 4
f1[22:25, 16:20] .= 3

f2 = 2*ones(n,m)
f2[4:7, 15:20] .= 4
f2[14:18, 30:36] .= 7
f2[22:25, 12:18] .= 5

allsrc = DataFrame(x=[],y=[])
allrcv = DataFrame(x=[],y=[])
for i = 1:50
    push!(allsrc.x,rand(1:m))
    push!(allsrc.y,rand(1:n))
end
for i = 1:50
    push!(allrcv.x,rand(1:m))
    push!(allrcv.y,rand(1:n))
end
numsrc = size(allsrc,1)
numrcv = size(allrcv,1)

u1 = PyObject[]; u2 = PyObject[]
for i = 1:numsrc
    push!(u1, eikonal(f1,allsrc.x[i],allsrc.y[i],h))
    push!(u2, eikonal(f2,allsrc.x[i],allsrc.y[i],h))
end
sess = Session()
init(sess)
uobs1 = run(sess, u1)
uobs2 = run(sess, u2)

obs_time1 = Array{Float64}(undef,numsrc,numrcv)
obs_time2 = Array{Float64}(undef,numsrc,numrcv)
for i = 1:numsrc
    for j = 1:numrcv
        obs_time1[i,j] = uobs1[i][allrcv.y[j],allrcv.x[j]]
        obs_time2[i,j] = uobs2[i][allrcv.y[j],allrcv.x[j]]
    end
end

folder = "/home/lingxia/ADTomo.jl/tests/readin_data/2D_test/"
h5write(folder * "obs_1.h5","data",obs_time1)
h5write(folder * "obs_2.h5","data",obs_time2)
CSV.write(folder * "allsrc.csv", allsrc)
CSV.write(folder * "allrcv.csv", allcsv)