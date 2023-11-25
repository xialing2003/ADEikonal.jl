using CSV
using DataFrames

eve = CSV.read("seismic_data/BayArea/obspy/catalog.csv",DataFrame)

num = size(eve,1)

for i = 1:num-1
    for j = i+1:num
        if eve[i,4] == eve[j,4] && eve[i,5] == eve[j,5]
            print(eve[i,1],' ',eve[j,1],'\n')
        end
    end
end