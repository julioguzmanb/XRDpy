#!/bin/bash
#PBS -S /bin/bash
#PBS -q serial
#PBS -N xrdpy_single_shot_1d
#PBS -V
#PBS -J 1-20

set -euo pipefail

if [ -n "${XRDPY_ENV_SETUP:-}" ]; then
    source "${XRDPY_ENV_SETUP}"
fi

: "${XRDPY_PATH_ROOT:?Set XRDPY_PATH_ROOT to the SACLA experiment root.}"
: "${XRDPY_METADATA_H5:?Set XRDPY_METADATA_H5 to the selection HDF5.}"
: "${XRDPY_PONI_PATH:?Set XRDPY_PONI_PATH to the calibrated PONI file.}"

XRDPY_ANALYSIS_SUBDIR="${XRDPY_ANALYSIS_SUBDIR:-analysis}"
XRDPY_RAW_SUBDIR="${XRDPY_RAW_SUBDIR:-}"
XRDPY_MASK_PATH="${XRDPY_MASK_PATH:-}"
XRDPY_N_CHUNKS="${XRDPY_N_CHUNKS:-20}"
XRDPY_BEAMLINE="${XRDPY_BEAMLINE:-}"
XRDPY_DETECTOR_ID="${XRDPY_DETECTOR_ID:-MPCCD-8N0-3-002}"
XRDPY_AZIMUTHAL_EDGES="${XRDPY_AZIMUTHAL_EDGES:--90 -45 0 45 90}"
XRDPY_INCLUDE_FULL="${XRDPY_INCLUDE_FULL:-1}"
XRDPY_FULL_RANGE="${XRDPY_FULL_RANGE:--90 90}"
XRDPY_NPT="${XRDPY_NPT:-1000}"
XRDPY_AZIM_OFFSET_DEG="${XRDPY_AZIM_OFFSET_DEG:--90}"
XRDPY_THRESHOLD_COUNTS="${XRDPY_THRESHOLD_COUNTS:-40}"

TASK_ID="${PBS_ARRAYID:-${PBS_ARRAY_INDEX:-}}"
if [ -z "${TASK_ID}" ]; then
    echo "ERROR: This job is not running as a PBS array task."
    exit 1
fi

cd "${XRDPY_PATH_ROOT}"
mkdir -p logs
exec 1>"logs/${PBS_JOBNAME}.o${PBS_JOBID}.${TASK_ID}" \
     2>"logs/${PBS_JOBNAME}.e${PBS_JOBID}.${TASK_ID}"

read -r -a AZIMUTHAL_EDGES <<< "${XRDPY_AZIMUTHAL_EDGES}"
read -r -a FULL_RANGE <<< "${XRDPY_FULL_RANGE}"

OPTIONAL_ARGS=()
if [ -n "${XRDPY_MASK_PATH}" ]; then
    OPTIONAL_ARGS+=(--mask "${XRDPY_MASK_PATH}")
fi
if [ -n "${XRDPY_BACKGROUND:-}" ]; then
    OPTIONAL_ARGS+=(--background "${XRDPY_BACKGROUND}")
fi
if [ -n "${XRDPY_BACKGROUND_PATH:-}" ]; then
    OPTIONAL_ARGS+=(--background-path "${XRDPY_BACKGROUND_PATH}")
fi
if [ -n "${XRDPY_POLARIZATION_FACTOR:-}" ]; then
    OPTIONAL_ARGS+=(--polarization-factor "${XRDPY_POLARIZATION_FACTOR}")
fi
if [ -n "${XRDPY_INTENSITY_COL:-}" ]; then
    OPTIONAL_ARGS+=(--intensity-col "${XRDPY_INTENSITY_COL}")
fi
if [ -n "${XRDPY_BEAMLINE:-}" ]; then
    OPTIONAL_ARGS+=(--beamline "${XRDPY_BEAMLINE}")
fi
if [ "${XRDPY_INCLUDE_FULL}" = "0" ]; then
    OPTIONAL_ARGS+=(--no-include-full)
fi
if [ "${XRDPY_OVERWRITE:-0}" = "1" ]; then
    OPTIONAL_ARGS+=(--overwrite)
fi

python3 -m trxrdpy.analysis.Spring8_SACLA.single_shot_azimint \
    --metadata-h5 "${XRDPY_METADATA_H5}" \
    --poni "${XRDPY_PONI_PATH}" \
    --azimuthal-edges "${AZIMUTHAL_EDGES[@]}" \
    --full-range "${FULL_RANGE[@]}" \
    --npt "${XRDPY_NPT}" \
    --azim-offset-deg "${XRDPY_AZIM_OFFSET_DEG}" \
    --detector-id "${XRDPY_DETECTOR_ID}" \
    --threshold-counts "${XRDPY_THRESHOLD_COUNTS}" \
    --path-root "${XRDPY_PATH_ROOT}" \
    --raw-subdir "${XRDPY_RAW_SUBDIR}" \
    --analysis-subdir "${XRDPY_ANALYSIS_SUBDIR}" \
    --chunk "${TASK_ID}" \
    --n-chunks "${XRDPY_N_CHUNKS}" \
    "${OPTIONAL_ARGS[@]}"
