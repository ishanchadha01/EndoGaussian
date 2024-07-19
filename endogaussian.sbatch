#!/bin/bash
#SBATCH -J eg-c3vd                                                 # Job name
#SBATCH --account=gts-ychen3538                                 # Tracking account
#SBATCH -N2 --gres=gpu:A100:2                                   # Number of nodes and cores per node required
#SBATCH -t720                                                   # Duration of the job (Ex: 15 mins)
#SBATCH -qinferno                                               # QOS Name
#SBATCH -o eg-c3vd.out                                   # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL                              # Mail preferences
#SBATCH --mail-user=ichadha3@gatech.edu                         # E-mail address for notifications

cd /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian

module load anaconda3/2022.05                                   # Load module dependencies
module load cuda/11.7.0-7sdye3
module load gcc

conda activate endogaussian
python3.7 train.py -s /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data/cecum_t1_b --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py
python3.7 train.py -s /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data/cecum_t2_c --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py
python3.7 train.py -s /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data/cecum_t3_a --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py
python3.7 train.py -s /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data/cecum_t4_a --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py
python3.7 train.py -s /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data/sigmoid_t1_a --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py
python3.7 train.py -s /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data/sigmoid_t2_a --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py
python3.7 train.py -s /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data/sigmoid_t3_a --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py
python3.7 train.py -s /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data/sigmoid_t3_b --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py
python3.7 train.py -s /storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data/trans_t4_b --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py


# find "/storage/home/hcoda1/3/ichadha3/p-ychen3538-0/ishan/EndoGaussian/c3vd_data" -maxdepth 1 -mindepth 1 -type d | while read subdir; do
#     echo $subdir
#     python3.7 train.py -s $subdir --port 6017 --expname ct1a --configs arguments/endonerf/pulling.py
# done