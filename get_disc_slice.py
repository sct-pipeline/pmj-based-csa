#!/usr/bin/env python
# -*- coding: utf-8
# This scripts returns the slices (I-S) to compute CSA with a 3 slice extent from labels (discs or spinal rootlets)
# Author: Sandrine Bédard

import argparse
import numpy as np
import nibabel as nib
import pandas as pd
import os
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="Get the slices in (I-S) from label file with a 3 slice extent. Returns a list of the slice range. Example: 2:4")
    parser.add_argument('-label', required=True, type=str,
                        help="Nifti file of the disc or nerve labels.")
    parser.add_argument('-o', required=True, type=str,
                        help="Path to save labels corespondances.")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    # Read label file
    label = nib.load(args.label)
    # Get labels
    label_index = np.where(label.get_fdata() != 0)[-1]
    label = label.get_fdata()[np.where(label.get_fdata() != 0)]
    z = []
    i = 0
    log = pd.DataFrame(columns=['File', 'Level', 'Slices'])
    for label_idx in label_index:
        idx_low = label_idx - 1
        idz_high = label_idx + 1
        range = '{}:{}'.format(idx_low, idz_high)
        log = log.append({'File': args.label, 'Level': label[i], 'Slices': '{}:{}'.format(idx_low, idz_high)}, ignore_index=True)
        z.append(range)
        i = i + 1
    log.to_csv(os.path.join(os.path.abspath(args.o), args.label + '_labels.csv'))
    # Create a list of string with the ranges to use in process_data.sh
    returnStr = ''
    for item in z:
        returnStr += str(item)+' '
    print(returnStr)
    sys.exit(0)


if __name__ == '__main__':
    main()
