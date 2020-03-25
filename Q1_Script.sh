#!/bin/bash
#$ -l h_rt=2:00:00  #time needed
#$ -pe smp 2 #number of cores
#$ -l rmem=4G #number of memery
#$ -o Q1_Output.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M rkhussain1@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit Code/Q1_Code.py