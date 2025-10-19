#!/bin/bash -l
#PBS -N compute_ecr
#PBS -A UWAS0141
#PBS -l select=1:ncpus=128:mem=235GB
#PBS -l walltime=01:00:00
#PBS -q develop
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/compute_ecr.out

cd /glade/u/home/dstiller/dev/lmrecon

pixi run --frozen python lmrecon/scripts/compute_ecr.py 2024-07-15T16-08-15
