#!/bin/bash
#SBATCH  --output=/home/esandstroem/scratch/point-slam/output/log/%j.out
#SBATCH  --error=/home/esandstroem/scratch/point-slam/output/log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G
#SBATCH  --constraint='titan_xp' # geforce_rtx_2080_ti'

JOB_START_TIME=$(date)
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}" 
echo "Running on node: $(hostname)"
echo "Starting on:     ${JOB_START_TIME}" 
export PATH=/home/esandstroem/scratch/venvs/point_slam_env_github/bin:$PATH
source /home/esandstroem/scratch/point-slam/setup_paths.sh

datasets=("Replica" "TUM_RGBD" "ScanNet")
replica_scenes=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")
tum_scenes=("freiburg1_desk" "freiburg1_desk2" "freiburg1_room" "freiburg2_xyz" "freiburg3_office")
scannet_scenes=("scene0000" "scene0059" "scene0106" "scene0169" "scene0181" "scene0207" "scene0025" "scene0062" "scene0103" "scene0126")
output_affix="/home/esandstroem/scratch/point-slam/output"

method="point-slam"
dataset=${datasets[2]}
scene_name="scene0207"

run_args=""
run_suffix=""

# Run single or array job
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    python -u run.py configs/${dataset}/${scene_name}.yaml $run_args --output ${output_affix}/${dataset}/${scene_name}${run_suffix}
else
    scene_name=${replica_scenes[$SLURM_ARRAY_TASK_ID]} # start with 0
    python -u run.py configs/${dataset}/${scene_name}.yaml $run_args --output ${output_affix}/${dataset}/${scene_name}${run_suffix}
fi

# Send some noteworthy information to the output log
echo ""
echo "Job Comment:     test POINT-SLAM"
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     ${JOB_START_TIME}"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
