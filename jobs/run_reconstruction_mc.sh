#!/bin/bash -l
#PBS -N run_reconstruction_mc
#PBS -A UWAS0141
#PBS -l select=1:ncpus=2:mem=8GB
#PBS -l walltime=2:00:00
#PBS -q casper
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/run_reconstruction_mc_miroc.^array_index^.out
#PBS -J 1-20

cd /glade/u/home/dstiller/dev/lmrecon

pixi run --frozen python lmrecon/scripts/run_reconstruction_single.py 2026-02-23T14-16-28 MIROC-mc $PBS_ARRAY_INDEX
