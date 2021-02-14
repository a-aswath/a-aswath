#!/bin/bash                                       
#SBATCH --time=2-12:00
#SBATCH --job-name=my_first_slurm_job 
#SBATCH --mail-type=BEGIN,END,FAIL 
#SBATCH --mail-user=anusha.aswath@gmail.com  
#SBATCH --output=job-%j.log 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40:1 
source /data/p301644/python_venvs/mitochondria/bin/activate
python /home/p301644/mitochondria/RetinaVesselSegTrain1.py
