#!/bin/bash
#PBS -S /bin/bash
#PBS -q serial
#PBS -N delay_DET70_110K_1500nm_25p0mJ_40fs
#PBS -V
#PBS -J 1-20

cd /UserData/julioguzmanb/SACLA2024
source /home/julioguzmanb/main.bashrc

export XRDPY_PATH_ROOT="/UserData/julioguzmanb/SACLA2024"
export XRDPY_ANALYSIS_SUBDIR="analysis"
export XRDPY_TIME_METADATA_SUBDIR="TM_data"

mkdir -p logs

TASK_ID="$PBS_ARRAYID"
if [ -z "$TASK_ID" ]; then
    TASK_ID="$PBS_ARRAY_INDEX"
fi
if [ -z "$TASK_ID" ]; then
    echo "ERROR: This job is not running as a PBS array task (PBS_ARRAYID is empty)."
    exit 1
fi

exec 1>logs/${PBS_JOBNAME}.o${PBS_JOBID}.${TASK_ID} 2>logs/${PBS_JOBNAME}.e${PBS_JOBID}.${TASK_ID}

python3 -m XRDpy.analysis.Spring8_SACLA.datared \
  --mode final_imgs \
  --bl 3 \
  --chunk "${TASK_ID}" \
  --n_chunks 20 \
  --time_window_fs 40 \
  --scan_type delay \
  --background_run 1466639 \
  --sample_name DET70 \
  --temperature_K 110 \
  --excitation_wl_nm 1500 \
  --fluence_mJ_cm2 25 \
  --save_laser_off \
  --max_shots 50