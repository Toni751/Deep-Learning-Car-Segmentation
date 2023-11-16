#!/bin/sh
### –- specify queue --
-q gpuv100
### -- ask for number of cores (default: 1) --
-n 1
### -- Select the resources: 1 gpu in exclusive process mode --
-gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
-W 1:00
# request 5GB of system-memory
-R "rusage[mem=5GB]"



# Load environment variables
source ./.env


# Create job_out if it is not present
if [[ ! -d ${REPO}/job_out ]]; then
	mkdir ${REPO}/job_out
fi


date=$(date +%Y%m%d_%H%M)
mkdir ${REPO}/runs/train/${date}


# Activate venv
module load python3/3.10.12
module load cuda/12.1
source ${REPO}/venv/bin/activate


# run training
python3 carsegmentation.py --config config.yaml
