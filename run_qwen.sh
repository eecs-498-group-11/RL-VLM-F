#!/bin/bash 
# The interpreter used to execute the script
# Lines beginning with #SBATCH specify your computing resources and other logistics about how to run your job.

# Access computing resources allocated to the MLRE course account, the section may be 006 or 007 depending on the student.
#SBATCH --account=eecs498f25s006_class

# Specify the maximum runtime (in Hours:Minutes:Seconds). If your job hits that runtime, it will be terminated.
#SBATCH --time=4:00:00

# Specify whether to use a GPU (partition=gpu) or CPU (partition=standard). If you use the GPU partition, only request one GPU (gpus=1).
# Note that if you use the standard partition, you may have to remove the gpu configuration.
#SBATCH --partition=gpu
#SBATCH --gpus=1

# There are also more specific settings for configuring your CPU/GPU. You can reference the documentation for more information.
# Specify the amount of memory you need. If your job exceeds this memory limit, it will be terminated.
#SBATCH --mem=64g

# Name this job and the output file
#SBATCH --job-name=rlvlmf_qwen_test_1
#SBATCH --output=out_files/qwen_test_1.out

# Receive an email when your job starts and ends
#SBATCH --mail-user=pajehan@umich.edu
#SBATCH --mail-type=BEGIN,END

python3 qwen.py
