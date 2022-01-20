#!/bin/bash
#
# Process data to identify nerve rootlets of 3 neck positions.
#
# Usage:
#   ./identify_nerve.sh <SUBJECT>
#
# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/anat/
#
# Authors: Sandrine Bédard

set -x
# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params
SUBJECT=$1

# Save script path
PATH_SCRIPT=$PWD

# get starting time:
start=`date +%s`

# FUNCTIONS
# ==============================================================================

# Check if manual segmentation already exists. If it does, copy it locally. If it does not, perform segmentation.
segment_if_does_not_exist(){
  local file="$1"
  local contrast="$2"
  folder_contrast="anat"

  # Update global variable with segmentation file name
  FILESEG="${file}_seg"
  FILESEGMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/${folder_contrast}/${FILESEG}-manual.nii.gz"
  echo
  echo "Looking for manual segmentation: $FILESEGMANUAL"
  if [[ -e $FILESEGMANUAL ]]; then
    echo "Found! Using manual segmentation."
    rsync -avzh $FILESEGMANUAL ${FILESEG}.nii.gz
    sct_qc -i ${file}.nii.gz -s ${FILESEG}.nii.gz -p sct_deepseg_sc -qc ${PATH_QC} -qc-subject ${SUBJECT}
  else
    echo "Not found. Proceeding with automatic segmentation."
    # Segment spinal cord
    sct_deepseg_sc -i ${file}.nii.gz -c $contrast -qc ${PATH_QC} -qc-subject ${SUBJECT}
  fi
}

# Check if PMJ label exists. If it does not, perform automatic detection.
detect_pmj_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels-pmj"  # MAYBE TO CHANGE
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${file}_pmj.nii.gz
    sct_qc -i ${file}.nii.gz -s ${file}_pmj.nii.gz -p sct_detect_pmj -qc ${PATH_QC} -qc-subject ${SUBJECT}

  else
    echo "Not found. Proceeding with automatic labeling."
    # Detect PMJ
    sct_detect_pmj -i ${file}.nii.gz -c t2 -qc ${PATH_QC}
    #sct_detect_pmj -i ${file}.nii.gz -c t2 -s ${file_seg}.nii.gz -qc ${PATH_QC}

  fi
}



# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

SUBJECT_ID=$(dirname "$SUBJECT")

SES=$(basename "$SUBJECT")
# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy source images
mkdir -p ${SUBJECT}
rsync -avzh $PATH_DATA/$SUBJECT/ ${SUBJECT}
# Go to anat folder where all structural data are located
echo $PWD

cd ${SUBJECT}/anat/


file_t2="${SUBJECT_ID}_${SES}_T2w"

sct_image -i ${file_t2}.nii.gz -setorient RPI -o ${file_t2}.nii.gz

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t2 "t2"
file_t2_seg=$FILESEG
detect_pmj_if_does_not_exist $file_t2 $file_t2_seg
# process seg only to generate centerline
sct_process_segmentation -i ${file_t2_seg}.nii.gz -pmj ${file_t2}_pmj.nii.gz -pmj-distance 50 -o ${PATH_RESULTS}/csa-SC_pmj.csv -append 1 -qc ${PATH_QC} -qc-subject ${SUBJECT} -qc-image ${file_t2}.nii.gz -v 2

file_pmj_ctl=${file_t2}_seg_centerline_extrapolated
# Straigthen the spinal cord
sct_straighten_spinalcord -i ${file_t2}.nii.gz -s ${file_pmj_ctl}.nii.gz
file_t2_straight="${file_t2}_straight"

# Denoise
sct_maths -i ${file_t2_straight}.nii.gz -denoise 1 -o ${file_t2_straight}_denoised.nii.gz
file_t2_straight_denoised="${file_t2_straight}_denoised"

# Open FSLeyes to identify nerve rootlets
fsleyes ${file_t2_straight_denoised}.nii.gz

# Bring back label to curved space
file_t2_straight_nerve="${file_t2_straight_denoised}_spinalroots"
sct_apply_transfo -i ${file_t2_straight_nerve}.nii.gz -w warp_straight2curve.nii.gz -d ${file_t2}.nii.gz -x label -o ${file_t2}_labels-spinalroots-manual.nii.gz

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
  "${SUBJECT_ID}_${SES}_T2w_labels-spinalroots-manual.nii.gz"
)
pwd
for file in ${FILES_TO_CHECK[@]}; do
  if [[ ! -e $file ]]; then
    echo "${SUBJECT}/anat/${file} does not exist" >> $PATH_LOG/_error_check_output_files.log
  fi
done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
