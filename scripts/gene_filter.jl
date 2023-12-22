using HDF5

sh1 = 5; sh2 = convert(Int64, (sh1+1)/2)
sv1 = 3; sv2 = convert(Int64, (sv1+1)/2)
fil_matrix = ones(sh1,sh1,sv1)

for i = 1:sh1
    for j = 1:sh1
        for k = 1:sv1
            fil_matrix[i,j,k] = exp(-((i-sh2)^2 + (j-sh2)^2 + (k-sv2)^2)/4)
        end
    end
end

fil_matrix = fil_matrix / sum(fil_matrix)
h5write("../local/BayArea/readin_data/filter/center.h5","data",fil_matrix)