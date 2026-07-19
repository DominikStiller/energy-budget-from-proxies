#!/bin/bash -l
#PBS -N compute_seasonal_anomalies_ecearth
#PBS -A UWAS0141
#PBS -l select=1:ncpus=32:mem=720GB
#PBS -l walltime=03:00:00
#PBS -q casper@casper-pbs
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/compute_seasonal_anomalies_ecearth.out

cd /glade/u/home/dstiller/dev/lmrecon


echo "Starting EC-Earth3-Veg-LR past1000"
pixi run --frozen python lmrecon/scripts/compute_seasonal_averages_ecearth.py \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py EC-Earth3-Veg-LR past1000 \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_detrended_anomalies.py EC-Earth3-Veg-LR past1000

echo "Starting EC-Earth3-Veg-LR historical"
pixi run --frozen python lmrecon/scripts/compute_seasonal_averages.py EC-Earth3-Veg-LR historical \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py EC-Earth3-Veg-LR historical

pixi run --frozen python lmrecon/scripts/compute_combined_seasonal_anomalies.py EC-Earth3-Veg-LR past1000 historical
