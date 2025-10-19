#!/bin/bash -l
#PBS -N run_reconstruction
#PBS -A UWAS0141
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=3:00:00
#PBS -q casper
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/run_reconstruction.out

cd /glade/u/home/dstiller/dev/lmrecon

echo
echo "Starting reconstruction (job ${PBS_JOBID})"

pixi run --frozen python lmrecon/scripts/run_reconstruction_allproxies.py obs
# pixi run --frozen python lmrecon/scripts/run_reconstruction_single.py obs id seed
