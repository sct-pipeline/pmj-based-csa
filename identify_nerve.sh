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
file_pmj_ctl="pmj_ctl/${file_t2}_seg_centerline_extrapolated"
# Straigten the spinal cord
sct_straighten_spinalcord -i ${file_t2}.nii.gz -c t2 -s ${file_pmj_ctl}.nii.gz
file_t2_straight="${file_t2}_straight"
# Smooth along AP direction
sct_maths -smooth 0,1,0 -i ${file_t2_straight}.nii.gz -o ${file_t2_straight}_smooth.nii.gz
file_t2_straight_smooth="${file_t2_straight}_smooth"
# Symmetrize along RL
sct_maths -i ${file_t2_straight_smooth}.nii.gz -symmetrize 0 -o ${file_t2_straight_smooth}_sym.nii.gz
