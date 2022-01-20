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

#SUBJECT_ID=$(dirname "$SUBJECT")

#SES=$(basename "$SUBJECT")
# Go to folder where data will be copied and processed
cd ${PATH_DATA_PROCESSED}
# Copy source images
mkdir -p ${SUBJECT}
rsync -avzh $PATH_DATA/$SUBJECT/ ${SUBJECT}
# Go to anat folder where all structural data are located
echo $PWD

cd ${SUBJECT}/anat/

# Head Normal
ses="ses-headNormal"
file_t2_normal="${SUBJECT}_${ses}_T2w"

sct_image -i ${file_t2_normal}.nii.gz -setorient RPI -o ${file_t2_normal}.nii.gz

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t2_normal "t2"
file_t2_normal_seg=$FILESEG
detect_pmj_if_does_not_exist $file_t2_normal $file_t2_seg
# process seg only to generate centerline
sct_process_segmentation -i ${file_t2_normal_seg}.nii.gz -pmj ${file_t2_normal}_pmj.nii.gz -pmj-distance 50 -o ${PATH_RESULTS}/csa-SC_pmj.csv -append 1 -qc ${PATH_QC} -qc-subject ${SUBJECT} -qc-image ${file_t2_normal}.nii.gz -v 2

file_pmj_ctl=${file_t2_normal}_seg_centerline_extrapolated
# Straigthen the spinal cord
# TODO: add -ofolder to save different warping fields
sct_straighten_spinalcord -i ${file_t2_normal}.nii.gz -s ${file_pmj_ctl}.nii.gz
file_t2_normal_straight="${file_t2_normal}_straight"
# Smooth along AP direction

#sct_maths -smooth 0,1,0 -i ${file_t2_normal_straight}.nii.gz -o ${file_t2_normal_straight}_smooth.nii.gz
#file_t2_normal_straight_smooth="${file_t2_normal_straight}_smooth"
file_t2_normal_straight_smooth=${file_t2_normal_straight}
# Symmetrize along RL
sct_maths -i ${file_t2_normal_straight_smooth}.nii.gz -symmetrize 0 -o ${file_t2_normal_straight_smooth}_sym.nii.gz


# Head Up

ses="ses-headUp"
file_t2_up="${SUBJECT}_${ses}_T2w"

sct_image -i ${file_t2_up}.nii.gz -setorient RPI -o ${file_t2_up}.nii.gz

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t2_up "t2"
file_t2_up_seg=$FILESEG
detect_pmj_if_does_not_exist $file_t2_up $file_t2_seg
# process seg only to generate centerline
sct_process_segmentation -i ${file_t2_up_seg}.nii.gz -pmj ${file_t2_up}_pmj.nii.gz -pmj-distance 50 -o ${PATH_RESULTS}/csa-SC_pmj.csv -append 1 -qc ${PATH_QC} -qc-subject ${SUBJECT} -qc-image ${file_t2_up}.nii.gz -v 2

file_pmj_ctl=${file_t2_up}_seg_centerline_extrapolated
# Straigthen the spinal cord
# TODO: add -ofolder to save different warping fields
sct_straighten_spinalcord -i ${file_t2_up}.nii.gz -s ${file_pmj_ctl}.nii.gz
file_t2_up_straight="${file_t2_up}_straight"

# Smooth along AP direction

#sct_maths -smooth 0,1,0 -i ${file_t2_up_straight}.nii.gz -o ${file_t2_up_straight}_smooth.nii.gz
#file_t2_up_straight_smooth="${file_t2_up_straight}_smooth"
file_t2_up_straight_smooth=${file_t2_up_straight}
# Symmetrize along RL
sct_maths -i ${file_t2_up_straight_smooth}.nii.gz -symmetrize 0 -o ${file_t2_up_straight_smooth}_sym.nii.gz


# Head Down

ses="ses-headDown"

file_t2_down="${SUBJECT}_${ses}_T2w"

sct_image -i ${file_t2_down}.nii.gz -setorient RPI -o ${file_t2_down}.nii.gz

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t2_down "t2"
file_t2_down_seg=$FILESEG
detect_pmj_if_does_not_exist $file_t2_down $file_t2_seg
# process seg only to generate centerline
sct_process_segmentation -i ${file_t2_down_seg}.nii.gz -pmj ${file_t2_down}_pmj.nii.gz -pmj-distance 50 -o ${PATH_RESULTS}/csa-SC_pmj.csv -append 1 -qc ${PATH_QC} -qc-subject ${SUBJECT} -qc-image ${file_t2_down}.nii.gz -v 2

file_pmj_ctl=${file_t2_down}_seg_centerline_extrapolated
# Straigthen the spinal cord
# TODO: add -ofolder to save different warping fields
sct_straighten_spinalcord -i ${file_t2_down}.nii.gz -s ${file_pmj_ctl}.nii.gz
file_t2_down_straight="${file_t2_down}_straight"
# Smooth along AP direction

#sct_maths -smooth 0,1,0 -i ${file_t2_down_straight}.nii.gz -o ${file_t2_down_straight}_smooth.nii.gz
#file_t2_down_straight_smooth="${file_t2_down_straight}_smooth"
file_t2_down_straight_smooth=${file_t2_down_straight}
# Symmetrize along RL
sct_maths -i ${file_t2_down_straight_smooth}.nii.gz -symmetrize 0 -o ${file_t2_down_straight_smooth}_sym.nii.gz

# Register to headNormal
sct_register_multimodal -i ${file_t2_up_straight_smooth}_sym.nii.gz -d ${file_t2_normal_straight_smooth}_sym.nii.gz -param step=1,type=im,metric=MeanSquares,algo=affine,iter=15:step=2,type=im,metric=MeanSquares,algo=syn,iter=10,shrink=2 -qc ${PATH_QC} -qc-subject ${SUBJECT}
sct_register_multimodal -i ${file_t2_down_straight_smooth}_sym.nii.gz -d ${file_t2_normal_straight_smooth}_sym.nii.gz -param step=1,type=im,metric=MeanSquares,algo=affine,iter=15:step=2,type=im,metric=MeanSquares,algo=syn,iter=10,shrink=2 -qc ${PATH_QC} -qc-subject ${SUBJECT}


# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
#  "${SUBJECT_ID}_${SES}_T2w_seg.nii.gz" 
#  "${SUBJECT_ID}_${SES}_T2w_seg_labeled.nii.gz"
#  "${SUBJECT_ID}_${SES}_T2w_labels-spinalroots-manual.nii.gz"
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
