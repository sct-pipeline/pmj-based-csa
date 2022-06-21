#!/usr/bin/env python
# -*- coding: utf-8
# TODO
# Author: Sandrine Bédard

import argparse
import pandas as pd
import sys
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        description="TODO")
    parser.add_argument('-file-distance', required=True, type=str,
                        help=".csv file of the distance.")
    parser.add_argument('-subject', required=True, type=str,
                        help="subject ID. Example: sub-004")

    return parser


def csv2dataFrame(filename):
    """
    Loads a .csv file and builds a pandas dataFrame of the data
    Args:
        filename (str): filename of the .csv file
    Returns:
        data (pd.dataFrame): pandas dataframe of the .csv file's data
    """
    data = pd.read_csv(filename)
    return data


def main():
    parser = get_parser()
    args = parser.parse_args()

    dist = csv2dataFrame(args.file_distance).set_index('Subject')
    sub1 = args.subject + "_ses-headNormal"
    sub2 = args.subject + "_ses-headDown"
    sub3 = args.subject + "_ses-headUp"
    subs = [sub1, sub2, sub3]
    dist.drop(columns=['Distance - Disc (mm)', 'Disc'], inplace=True)
    distance_nerve = dist.loc[dist.index.intersection(subs)]
    distance_nerve.sort_values(by=['Nerve'], inplace=True)
    distance_nerve.dropna(inplace=True)
    mean = distance_nerve.groupby('Nerve').mean()
    returnStr = ''
    for item in mean['Distance - PMJ (mm)']:
        returnStr += str(np.round(item, 1))+' '
    print(returnStr)
    sys.exit(0)


if __name__ == '__main__':
    main()
