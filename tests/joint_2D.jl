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

m = 30; n = 40; h = 1.
f1 = ones(m,n)
f1[3:7, 5:10] .= 2
f1[14:18, 20:26] .= 4
f1[22:25, 16:20] .= 3
f2 = 2*ones(m,n)
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
        obs_time1[i,j] = uobs1[i][allrcv.x[j],allrcv.y[j]]
        obs_time2[i,j] = uobs2[i][allrcv.x[j],allrcv.y[j]]
    end
end

fvar = Variable(ones(m, n)) ; pvs  = Variable(2.)                       #design an original velocity model for inversion
u1 = PyObject[]; u2 = PyObject[]
for i=1:numsrc
    push!(u1,eikonal(fvar,allsrc.x[i],allsrc.y[i],h))
    push!(u2,eikonal(fvar*pvs,allsrc.x[i],allsrc.y[i],h))
end

loss1 = sum([sum((obs_time1[i,j] - u1[i][allrcv.x[j],allrcv.y[j]])^2) for i = 1:numsrc for j=1:numrcv])
loss2 = sum([sum((obs_time2[i,j] - u2[i][allrcv.x[j],allrcv.y[j]])^2) for i = 1:numsrc for j=1:numrcv])
loss = loss1+loss2

init(sess)
loss = mpi_sum(loss)
options = Optim.Options(iterations = 100)
result = ADTomo.mpi_optimize(sess,loss,method="LBFGS",options=options,loc="/home/lingxia/1/",steps= 1000)

if mpi_rank()==0
    @info [size(result[i]) for i = 1:length(result)]
    @info [length(result)]
end
mpi_finalize()
