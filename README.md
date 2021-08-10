# PMJ-based CSA
CSA measure based on distance from pontomedullary junction (PMJ)

# Analysis
## Neck positions
To compute CSA with vertebral-base CSA and PMJ-based CSA using method 1 of centerline extrapolation, run:
~~~
cd pmj-based-csa
sct-run_batch -jobs -1 -path-data <PATH_DATA> -path-out ~/pmj-based-cas_results-method-1 -script process_data_neck_position.sh
~~~

For method 2: 
~~~
cp csa_pmj_extrapolation.py $SCT_DIR/spinalcordtoolbox/csa_pmj.py
sct-run_batch -jobs -1 -path-data <PATH_DATA> -path-out ~/pmj-based-cas_results-method-2 -script process_data_neck_position.sh
~~~
