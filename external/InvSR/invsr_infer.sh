#!/bin/bash

#SBATCH --job-name=COCO_ENHANCE_val2017   
#SBATCH --output=slurm/COCO_ENHANCE_val2017/%a.out      
#SBATCH --error=slurm/COCO_ENHANCE_val2017/%a.err        
#SBATCH --time=10:00:00                
#SBATCH --mem=18G                       
#SBATCH -A uva_cv_lab
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=a6000
#SBATCH --mail-user=snf4za@virginia.edu  
#SBATCH --mail-type=BEGIN,END,FAIL    

module load cuda/12.2.2
module load gcc/11
module load ninja

python inference_invsr_us.py