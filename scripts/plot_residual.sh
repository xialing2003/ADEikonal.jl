#!/bin/bash

# this part generate the data for plot
cd ../local/BayArea/readin_data/
awk -F ',' 'NR > 1 {print $4 " " $5}' sta_eve/allsta.csv |
paste -d ' ' - for_S/residual/sta_ratio_s.txt > for_S/residual/plot_sta_ratio_s.txt

# this part use GMT to plot useful figures
# cd ../gmt_plot
# gmt begin ratio_p png
#     gmt basemap -R-123.5/-120.5/36/38.6 -JM15c -Ba
#     gmt grdimage @earth_relief_01m -Baf -BWSen -t50
#     gmt colorbar
#     gmt makecpt -Cpolar -T-1/1
#     gmt plot -Sc0.2c -C ../readin_data/for_P/residual/plot_sta_ratio_p.txt

#     gmt colorbar -DjTR+w3i/0.2i+o0.5c/0.5c+v+m -C -Bxa0.4f0.1 -By+l"ratio" 
# gmt end