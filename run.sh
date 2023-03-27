#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --partition=Teach-LongJobs
#SBATCH --gres=gpu:2
#SBATCH --mem=0  # memory in Mb
#SBATCH --time=0-08:00:00

export CUDA_HOME=/opt/cuda-11.2.0/

export CUDNN_HOME=/opt/cudnn-11.4-8.2.2.26/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

# Activate the relevant virtual environment:


source $HOME/miniconda3/bin/activate mlp2

export LS=$(ls /opt/)

echo $LS

export NVIDIA_SMI=$(nvidia-smi)

echo $NVIDIA_SMI

python abstractivesummarisation.py
