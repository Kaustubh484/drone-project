#!/bin/bash
#SBATCH --job-name=drone_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu           # Check 'sinfo' for correct partition name
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# 1. Load Apptainer
module load apptainer

# 2. Go to your code folder (where train_drone.py is)
cd /fs/classhomes/kshah115/px4-drone-navigation


# 3. Pull the Image (Optional but good for caching)
# This converts your Docker image to a .sif file on the cluster
if [ ! -f drone_env.sif ]; then
    echo "Pulling container image..."
    apptainer pull drone_env.sif docker://kaustubh484/drone-env:latest
fi

# 4. Run Training
echo "Starting training..."
# We pass 'train_drone.py' and args because ENTRYPOINT is 'python'
apptainer run --nv drone_env.sif train_drone.py --algo sac --episodes 10000 --headless