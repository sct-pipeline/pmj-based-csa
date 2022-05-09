#!/usr/bin/env python
# -*- coding: utf-8
# This scripts returns the slices to compute CSA with a 3 slice extent from labels
# Author: Sandrine Bédard

import argparse
import csv
import numpy as np
import nibabel as nib
import os
import sys
import logging

FNAME_LOG= 'log_label_slices.txt'


# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)



def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute distance between C2-C3 intervertebral disc and PMJ. Outputs .csv file with results. | Orientation needs to be RPI")
    parser.add_argument('-label', required=True, type=str,
                        help="Nifti file of the vertebral or nerve labels.")
    parser.add_argument('-o', required=False, type=str,
                        default='pmj_disc_distance.csv',
                        help="Output csv filename.")

    return parser

def save_Nifti1(data, original_image, filename):
    empty_header = nib.Nifti1Header()
    image = nib.Nifti1Image(data, original_image.affine, empty_header)
    nib.save(image, filename)

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(os.path.abspath(os.curdir), FNAME_LOG))
    logging.root.addHandler(fh)
    #logger.info('{}'.format(args.label))

    label = nib.load(args.label)
    # Get discs labels
    label_index = np.where(label.get_fdata() != 0)[-1]
    label = label.get_fdata()[np.where(label.get_fdata() != 0)]
    z = []
    i = 0
    for label_idx in label_index:
        idx_low = label_idx - 1
        idz_high = label_idx + 1
        range = '{}:{}'.format(idx_low, idz_high)
        #logger.info('{}, {}:{}'.format(label[i], idx_low, idz_high))
        z.append(range)
        i = i + 1
    import sys
    returnStr = ''
    for item in z:
        returnStr += str(item)+' '
    print(returnStr)
    sys.exit(0)


if __name__ == '__main__':
    main()
    
