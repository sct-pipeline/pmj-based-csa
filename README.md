# PMJ-based CSA
CSA measure based on distance from pontomedullary junction (PMJ)


# Manual labeling of spinal cord rootlets with FSLeyes
## Installation

Download this repository:
```
git clone https://github.com/sct-pipeline/pmj-based-csa.git
```
## Run manual labeling script
1. In the terminal, go into this repository
~~~
cd pmj-based-csa
~~~
2. Run the correction script specifying `path-data` and `path-output`
~~~
sct_run_batch -jobs -1 -path-data <PATH-DATA> -path-output ~/pmj_csa_nerve_roots_results -script identify_nerve.sh
~~~

An FSLeyes window will open and the following steps are in FSLeyes.

4. Click on: *Tools → Edit mode*
5. Click on: *Edit (ortho view) → Create mask*
6. Change parameter `Fill value` according to the spinal level : 
7. Locate the C3 spinal root using the sagittal and axial views (usually around C2 vertebral level)
8. Identify the slices that cover the C3 spinal root using the coronal and axial views (see schematic below)
9. Place the label 3 at the center of the spinal cord on the median axial slice that cover the spinal root
10. Erase the 3 voxels we don’t want (verify that the label is only one voxel
11. Repeat steps 3 to 5 for each spinal root by modifying the parameter `Fill value`:
    * C4 → 4, C5 → 5, … , C8 → 8, T1 → 9, …, T4 → 12, …
    * Stop when the contrast is not strong enough to accurately label the spinal roots
12. Save the mask under the same folder than the image with the suffix `_spinalroots.nii.gz`

![image](https://user-images.githubusercontent.com/71230552/141651001-f0c438d7-ae1e-44ba-b689-c5f5b319be22.png)

13. Repeat step 4 to 12 for every image (the FSLeyes window will automatically open)

# Analysis pipeline

### Dependencies
* SCT
* Python 3.7

### Installation
Download this repository:
~~~
git clone https://github.com/sct-pipeline/pmj-based-csa
~~~

## Neck positions analysis
To compute CSA with vertebral-base CSA and PMJ-based CSA using method 1 of centerline extrapolation, run:
~~~
cd pmj-based-csa
sct_run_batch -jobs -1 -path-data <PATH_DATA> -path-out ~/pmj-based-cas_results-method-1 -script process_data_neck_position.sh
~~~

For method 2: 
~~~
cp csa_pmj_extrapolation.py $SCT_DIR/spinalcordtoolbox/csa_pmj.py
sct-run_batch -jobs -1 -path-data <PATH_DATA> -path-out ~/pmj-based-cas_results-method-2 -script process_data_neck_position.sh
~~~

To remove the added file of your current SCT, run the following line:
~~~
cd $SCT_DIR
git restore spinalcordtoolbox/csa_pmj.py
~~~
