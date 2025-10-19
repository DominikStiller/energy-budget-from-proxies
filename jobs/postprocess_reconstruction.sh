#!/bin/bash -l
#PBS -N postprocess_reconstruction
#PBS -A UWAS0141
#PBS -l select=1:ncpus=36:mem=720GB
#PBS -l walltime=05:00:00
#PBS -q casper
#PBS -l job_priority=economy
#PBS -j oe
#PBS -m abe
#PBS -o /glade/campaign/univ/uwas0141/lmrecon/job_output/postprocess_reconstruction.out

cd /glade/u/home/dstiller/dev/lmrecon

pixi run --frozen python lmrecon/scripts/postprocess_reconstruction.py 2025-02-24T22-24-27
