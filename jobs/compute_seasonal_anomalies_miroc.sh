#!/bin/bash -l
#PBS -N compute_seasonal_anomalies_miroc
#PBS -A UWAS0141
#PBS -l select=1:ncpus=32:mem=720GB
#PBS -l walltime=03:00:00
#PBS -q casper@casper-pbs
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/compute_seasonal_anomalies_miroc.out

cd /glade/u/home/dstiller/dev/lmrecon


echo "Starting MIROC-ES2L past1000"
pixi run --frozen python lmrecon/scripts/compute_seasonal_averages.py MIROC-ES2L past1000 lmrecon \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py MIROC-ES2L past1000 \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_detrended_anomalies.py MIROC-ES2L past1000

echo "Starting MIROC-ES2L historical"
pixi run --frozen python lmrecon/scripts/compute_seasonal_averages.py MIROC-ES2L historical lmrecon r1i1000p1f2 \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py MIROC-ES2L historical

pixi run --frozen python lmrecon/scripts/compute_combined_seasonal_anomalies.py MIROC-ES2L past1000 historical
