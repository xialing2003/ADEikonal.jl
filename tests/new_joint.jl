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
using Optim
using LineSearches
reset_default_graph()

mpi_init()
m = 40; n = 30
fvar_ = Variable(ones(n,m)); pvs_ = Variable(2.)
vari = vcat(tf.reshape(fvar_, (-1,)), tf.reshape(pvs_, (-1,)))
varn = mpi_bcast(vari)
fvar = tf.reshape(varn[1:prod(size(fvar_))], size(fvar_))
pvs = tf.reshape(varn[prod(size(fvar_))+1:end], size(pvs_))

rank = mpi_rank()
nproc = mpi_size()

h = 1.0          #resolution

folder = "/home/lingxia/ADTomo.jl/local/2Dtests/"
allsrc = CSV.read(folder * "allsrc.csv",DataFrame)
allrcv = CSV.read(folder * "allrcv.csv",DataFrame)
obs_time1 = h5read(folder * "obs_1.h5","data")
obs_time2 = h5read(folder * "obs_2.h5","data")
numsrc = size(allsrc,1); numrcv = size(allrcv,1)

allsrc = allsrc[rank+1:nproc:numsrc, :]
obs_time1 = obs_time1[rank+1:nproc:numsrc, :]
obs_time2 = obs_time2[rank+1:nproc:numsrc, :]
numsrc = size(allsrc, 1)

u1 = PyObject[]; u2 = PyObject[]
for i=1:numsrc
    push!(u1,eikonal(fvar,allsrc.x[i],allsrc.y[i],h))
    push!(u2,eikonal(fvar*pvs,allsrc.x[i],allsrc.y[i],h))
end

sum_loss_time = PyObject[]
for i = 1:numsrc
    for j = 1:numrcv
        push!(sum_loss_time, (obs_time1[i,j] - u1[i][allrcv.y[j],allrcv.x[j]])^2)
        push!(sum_loss_time, (obs_time2[i,j] - u2[i][allrcv.y[j],allrcv.x[j]])^2)
    end
end
sess = Session(intra=2)
run(sess, tf.global_variables_initializer()) 
local_loss = sum(sum_loss_time)

loss = mpi_sum(local_loss)
@show run(sess,loss, fvar=> ones(n,m),pvs=>2.0)
#=
vars = get_collection()
g1 = gradients(loss,vars[1])
g2 = gradients(loss,vars[2])
print("qs\n")
vs = vcat([tf.reshape(v, (-1,)) for v in vars]...)
x0 = run(sess, vs)
pl = placeholder(x0)
print("1\n")
r = mpi_rank()

grads1 = run(sess,g1)
grads2 = run(sess,g2)
print("5")
e = vec(permutedims(grads1)); push!(e,grads2)
print(e)
=#
options = Optim.Options(iterations = 100)
result = ADTomo.mpi_optimize(sess,loss,method="LBFGS",options=options,loc="/home/lingxia/1/",steps= 1000)

print(rank)
if rank==0
    @info [size(result[i]) for i = 1:length(result)]
    @info [length(result)]
end
#
mpi_finalize()