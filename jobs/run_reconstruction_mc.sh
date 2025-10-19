#!/bin/bash -l
#PBS -N run_reconstruction_mc
#PBS -A UWAS0141
#PBS -l select=1:ncpus=4:mem=64GB
#PBS -l walltime=10:00:00
#PBS -q casper
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/run_reconstruction_mc_mri.^array_index^.out
#PBS -J 1-20

cd /glade/u/home/dstiller/dev/lmrecon

pixi run --frozen python lmrecon/scripts/run_reconstruction_single.py 2025-04-04T18-11-01 MRI-mc $PBS_ARRAY_INDEX
