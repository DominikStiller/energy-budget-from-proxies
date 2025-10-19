#!/bin/bash -l
#PBS -N compute_seasonal_anomalies_mpi
#PBS -A UWAS0141
#PBS -l select=1:ncpus=32:mem=720GB
#PBS -l walltime=03:00:00
#PBS -q casper@casper-pbs
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/compute_seasonal_anomalies_mpi.out

cd /glade/u/home/dstiller/dev/lmrecon


echo "Starting MPI-ESM1-2-LR past2k"
pixi run --frozen python lmrecon/scripts/compute_seasonal_averages.py MPI-ESM1-2-LR past2k glade \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py MPI-ESM1-2-LR past2k \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_detrended_anomalies.py MPI-ESM1-2-LR past2k

echo "Starting MPI-ESM1-2-LR historical"
pixi run --frozen python lmrecon/scripts/compute_seasonal_averages.py MPI-ESM1-2-LR historical lmrecon r1i2000p1f1 \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py MPI-ESM1-2-LR historical

pixi run --frozen python lmrecon/scripts/compute_combined_seasonal_anomalies.py MPI-ESM1-2-LR past2k historical
