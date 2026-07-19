#!/bin/bash -l
#PBS -N compute_seasonal_anomalies_canesm
#PBS -A UWAS0141
#PBS -l select=1:ncpus=32:mem=720GB
#PBS -l walltime=03:00:00
#PBS -q casper@casper-pbs
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/compute_seasonal_anomalies_canesm.out

cd /glade/u/home/dstiller/dev/lmrecon


echo "Starting CanESM5 hist-nat"
pixi run --frozen python lmrecon/scripts/compute_seasonal_averages.py CanESM5 hist-nat \
  && pixi run --frozen python lmrecon/scripts/compute_seasonal_anomalies.py CanESM5 hist-nat
 
echo "Inferring F and R for CanESM5 hist-nat"
pixi run --frozen python lmrecon/scripts/compute_historical_forcing_and_response.py CanESM5 hist-nat r1i1p2f1,r2i1p2f1,r2i1p2f1
