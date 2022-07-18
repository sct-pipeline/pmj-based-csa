#!/usr/bin/env python
# -*- coding: utf-8
# Create BIDS dataset from nifti
# To convert from DICOM to nifti: dcm2niix -f %p_%s -o pmj_00X -z y pmj_009_dicom/
# Author: Sandrine Bédard

import argparse
import sys
import os
import shutil


BIDS_DICT = {'T2w_0.6mm_headDown_2.nii.gz': 'ses-headDown_T2w.nii.gz',
                            'T2w_0.6mm_headDown_2.json': 'ses-headDown_T2w.json',
                            'T2w_0.6mm_headNormal_4.nii.gz': 'ses-headNormal_T2w.nii.gz',
                            'T2w_0.6mm_headNormal_4.json': 'ses-headNormal_T2w.json', 
                            'T2w_0.6mm_headUp_6.nii.gz': 'ses-headUp_T2w.nii.gz',
                            'T2w_0.6mm_headUp_6.json': 'ses-headUp_T2w.json', 
                            }


def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert PMJ data to BIDS compatible dataset" )
    parser.add_argument('-path-in', required=True, type=str,
                        help="Path to converted nifti with all subjects.")
    parser.add_argument('-path-out', required=True, type=str,
                        help="Path to output BIDS dataset, which contains all the 'sub-' folders.")
    return parser

def main():
    # Parse input arguments
    parser = get_parser()
    args = parser.parse_args()
    if os.path.exists(args.path_out):
         shutil.rmtree (args.path_out)
    os.makedirs (args.path_out)
    path_out = os.path.abspath(args.path_out)
    path_in = os.path.abspath(args.path_in)
    list_subjects = [x for x in os.listdir(path_in) if os.path.isdir(os.path.join(path_in,x))]
    for subject in list_subjects:
        print(subject)
        for contrast in BIDS_DICT.keys():
            filename_archive = os.path.join(path_in, subject, contrast)
            ses = BIDS_DICT[contrast].split('_')[0]
            subject_id = 'sub-'+ subject[-3::]
            filename_BIDS = os.path.join(path_out, subject_id, ses, 'anat')
            if not os.path.exists(filename_BIDS):
                os.makedirs(filename_BIDS)
            os.system('cp ' + filename_archive + ' ' + filename_BIDS+ '/'+ subject_id + '_' + BIDS_DICT[contrast])

if __name__ == '__main__':
    main()