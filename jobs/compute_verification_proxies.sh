#!/bin/bash -l
#PBS -N run_reconstruction_mc
#PBS -A UWAS0141
#PBS -l select=1:ncpus=8:mem=64GB
#PBS -l walltime=3:00:00
#PBS -q casper
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/compute_verification_proxies.out
#PBS -J 1-5

cd /glade/u/home/dstiller/dev/lmrecon

pixi run --frozen python lmrecon/scripts/compute_verification_proxies.py MPI-mc1/$PBS_ARRAY_INDEX
