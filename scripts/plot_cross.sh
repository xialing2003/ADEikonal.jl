#!/bin/bash

cd ../local/BayArea/gmt_plot/
gmt begin cross_x png

    gmt basemap -R-124/-119.8/35.3/39.3 -JM15c -Ba -t50
    gmt plot ../seismic_data/SHP/ca_offshore.shp -W0.3p,lightblue4
    gmt plot ../seismic_data/SHP/fault_areas.shp -W0.3p,lightblue4
    gmt plot ../seismic_data/SHP/Qfaults_US_Database.shp -W0.3p,lightblue4
    gmt plot -W2p,royalblue -L -l"Range" << EOF
    -123.80712028585117 38.3731377818302
    -122.23763624860433 39.159851389716955
    -119.99992529287142 36.255171597377505
    -121.53301300000949 35.49760222861597
EOF
    gmt plot ../readin_data/plot/x.dat -W2p,darkblue,- -l"cross" 
    gmt plot ../readin_data/plot/mark_x.dat -St0.4c -Gred -l"mark"
    gmt text ../readin_data/plot/label_x.dat -F+f20p

gmt end