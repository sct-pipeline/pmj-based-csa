#!/bin/bash
#
# Process data of different neck positions (extension, flexion and straight).
#
# Usage:
#   ./process_data.sh <SUBJECT>
# 
# Manual segmentations and labels (discs, PMJ, nerve rootlets) should be located under:
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

# Check if manual label already exists. If it does, copy it locally. If it does
# not, perform labeling.
# NOTE: manual disc labels should go from C1-C2 to C7-T1.
label_if_does_not_exist(){
  local file="$1"
  local file_seg="$2"
  # Update global variable with segmentation file name
  FILELABEL="${file}_labels-disc"
  FILELABELMANUAL="${PATH_DATA}/derivatives/labels/${SUBJECT}/anat/${FILELABEL}-manual.nii.gz"
  echo "Looking for manual label: $FILELABELMANUAL"
  if [[ -e $FILELABELMANUAL ]]; then
    echo "Found! Using manual labels."
    rsync -avzh $FILELABELMANUAL ${FILELABEL}.nii.gz
    # Generate labeled segmentation from manual disc labels
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -discfile ${FILELABEL}.nii.gz -c t2
  else
    echo "Not found. Proceeding with automatic labeling."
    # Generate labeled segmentation
    sct_label_vertebrae -i ${file}.nii.gz -s ${file_seg}.nii.gz -c t2
  fi
}

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
  FILELABEL="${file}_labels-pmj"
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
cd ${SUBJECT}/anat/

file_t2="${SUBJECT_ID}_${SES}_T2w"

# Reorient to RPI
sct_image -i ${file_t2}.nii.gz -setorient RPI -o ${file_t2}.nii.gz

# Segment spinal cord (only if it does not exist)
segment_if_does_not_exist $file_t2 "t2"
file_t2_seg=$FILESEG

# Create labeled segmentation of vertebral levels (only if it does not exist) 
label_if_does_not_exist ${file_t2} ${file_t2_seg}
# Rename labeled segmentation and disc labels to differentiate with spinal levels (further down)
mv "${file_t2_seg}_labeled.nii.gz" "${file_t2_seg}_labeled_vert.nii.gz"
mv "${file_t2_seg}_labeled_discs.nii.gz" "${file_t2_seg}_labeled_discs_vert.nii.gz"
file_t2_seg_labeled="${file_t2_seg}_labeled_vert"

# Generate QC report to assess vertebral labeling
sct_qc -i ${file_t2}.nii.gz -s ${file_t2_seg_labeled}.nii.gz -p sct_label_vertebrae -qc ${PATH_QC} -qc-subject ${SUBJECT}
# Flatten scan along R-L direction (to make nice figures)
sct_flatten_sagittal -i ${file_t2}.nii.gz -s ${file_t2_seg}.nii.gz

# Detect PMJ
detect_pmj_if_does_not_exist $file_t2 $file_t2_seg

# Retereive nerve rootlet manual labels
file_nerve="${PATH_DATA}/derivatives/labels/${SUBJECT}/${folder_contrast}/${file_t2}_labels-spinalroots-manual"
if [[ -e $file_nerve ]]; then
  echo "Found! Using manual nerve rootlets labels."

else
  echo "Not found. Please provide manual nerve rootlet label using identify_nerve.sh."

fi

# Create a labeled segmentation with spinal levels
sct_label_vertebrae -i ${file_t2}.nii.gz -s ${file_t2_seg}.nii.gz -discfile ${file_nerve}.nii.gz -c t2 -qc ${PATH_QC} -qc-subject ${SUBJECT}
file_t2_seg_labeled_nerve="${file_t2_seg}_labeled"

# Get z slices of nerve labels
slices_nerves=($(python $PATH_SCRIPT/get_disc_slice.py -label ${file_t2_seg}_labeled_discs.nii.gz -o ${PATH_RESULTS}))
# Compute average cord CSA  at spinal levels
for slices_nerv in "${slices_nerves[@]}";do
  sct_process_segmentation -i ${file_t2_seg}.nii.gz -z $slices_nerv -o ${PATH_RESULTS}/${SUBJECT_ID}_${SES}_csa-SC_spinal.csv -append 1
done

# Run sct_process_segmentation to generate centerline
sct_process_segmentation -i ${file_t2_seg}.nii.gz -pmj ${file_t2}_pmj.nii.gz -pmj-distance 50 -pmj-extent 2 -o ${PATH_RESULTS}/${SUBJECT_ID}_${SES}_csa-50-pmj.csv -append 1 -qc ${PATH_QC} -qc-subject ${SUBJECT} -qc-image ${file_t2}.nii.gz -v 2
# Compute distance between PMJ and intervertebral discs to use as PMJ-based CSA distance.
python $PATH_SCRIPT/get_distance_pmj_disc.py -centerline ${file_t2_seg}_centerline_extrapolated.csv -disclabel ${file_t2_seg}_labeled_discs_vert.nii.gz -o ${PATH_RESULTS}/disc_pmj_distance.csv -spinalroots ${file_t2_seg}_labeled_discs.nii.gz -subject ${SUBJECT_ID}_${SES}

# Get slices to compute vertebral CSA
slices_vert=($(python $PATH_SCRIPT/get_disc_slice.py -label ${file_t2_seg}_labeled_discs_vert.nii.gz -o ${PATH_RESULTS}))
# Compute average cord CSA at every levels with 3 slices extent
for slices in "${slices_vert[@]}";do
  sct_process_segmentation -i ${file_t2_seg}.nii.gz -z $slices -o ${PATH_RESULTS}/${SUBJECT_ID}_${SES}_csa-SC_vert.csv -append 1
done

# Compute CSA perslice for graph
sct_process_segmentation -i ${file_t2_seg}.nii.gz -pmj ${file_t2}_pmj.nii.gz -perslice 1 -o ${PATH_RESULTS}/${SUBJECT_ID}_${SES}_perslice.csv -vertfile ${file_t2_seg}_labeled_vert.nii.gz -vert 1:10
# Generate graph of CSA as a function of PMJ distance
mkdir -p ${PATH_RESULTS}/graph_csa/
python $PATH_SCRIPT/generate_graph_csa_pmj.py -filename ${PATH_RESULTS}/${SUBJECT_ID}_${SES}_perslice.csv -o ${PATH_RESULTS}/graph_csa/${SUBJECT_ID}_${SES}.png

# Get average distance of nerve labels
dist_nerves_pmj=($(python $PATH_SCRIPT/get_mean_nerve_dist.py -file-distance ${PATH_SCRIPT}/disc_pmj_distance_to_use.csv -subject ${SUBJECT_ID}))
# Compute average cord CSA at distance from each nerve from PMJ with a 3 slice extent (or 1.8 mm)
for dist_nerv in "${dist_nerves_pmj[@]}";do
  sct_process_segmentation -i ${file_t2_seg}.nii.gz -pmj ${file_t2}_pmj.nii.gz -pmj-distance $dist_nerv -pmj-extent 1.8 -o ${PATH_RESULTS}/${SUBJECT_ID}_${SES}_csa-SC_pmj.csv -append 1 -qc ${PATH_QC} -qc-subject ${SUBJECT} -qc-image ${file_t2}.nii.gz -v 2
done



# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
  "${SUBJECT_ID}_${SES}_T2w_seg.nii.gz" 
  "${SUBJECT_ID}_${SES}_T2w_seg_centerline_extrapolated.csv"
  "${SUBJECT_ID}_${SES}_T2w_labeled_discs.nii.gz"  # Spinal rootlet labels
  "${SUBJECT_ID}_${SES}_T2w_labeled_discs_vert.nii.gz"  # Disc labels
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



