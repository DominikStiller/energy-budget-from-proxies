#!/bin/bash -l
#PBS -N compute_seasonal_anomalies_cesm
#PBS -A UWAS0141
#PBS -l select=1:ncpus=32:mem=720GB
#PBS -l walltime=03:00:00
#PBS -q casper@casper-pbs
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/compute_seasonal_anomalies_cesm.out

cd /glade/u/home/dstiller/dev/lmrecon


# echo "Starting CESM2 amip-piForcing"
# pixi run --frozen python lmrecon/scripts/compute_seasonal_averages_cesm.py \
#   /glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/f.e21.F1850_BGC.f09_f09_mg17.CFMIP-amip-piForcing.001 \
#   tas,rsdt,rsut,rlut
#   && pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py CESM2 amip-piForcing

# mv /glade/campaign/univ/uwas0141/lmrecon/cmip6/CESM2/f.e21.F1850_BGC.f09_f09_mg17.CFMIP-amip-piForcing.001 \
#   /glade/campaign/univ/uwas0141/lmrecon/cmip6/CESM2/amip-piForcing

# pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py CESM2 amip-piForcing


echo "Starting CESM2 past1000"
pixi run --frozen python lmrecon/scripts/compute_seasonal_averages_cesm.py \
  /glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002 \
  tas,tos,ohc300,ohc700,rsdt,rsut,rlut,siconc

mv /glade/campaign/univ/uwas0141/lmrecon/cmip6/CESM2/b.e21.BWmaHIST.f19_g17.PMIP4-past1000.002 \
  /glade/campaign/univ/uwas0141/lmrecon/cmip6/CESM2/past1000

pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py CESM2 past1000 \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_detrended_anomalies.py CESM2 past1000

echo "Starting CESM2 historical"
pixi run --frozen python lmrecon/scripts/compute_seasonal_averages_cesm.py \
  /glade/campaign/collections/cmip/CMIP6/timeseries-cmip6/b.e21.BWmaHIST.f19_g17.CMIP6-historical-WACCM-MA-2deg.003 \
  tas,tos,ohc300,ohc700,rsdt,rsut,rlut,siconc

mv /glade/campaign/univ/uwas0141/lmrecon/cmip6/CESM2/b.e21.BWmaHIST.f19_g17.CMIP6-historical-WACCM-MA-2deg.003 \
  /glade/campaign/univ/uwas0141/lmrecon/cmip6/CESM2/historical

pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py CESM2 historical

pixi run --frozen python lmrecon/scripts/compute_combined_seasonal_anomalies.py CESM2 past1000 historical
