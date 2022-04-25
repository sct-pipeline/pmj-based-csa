#!/usr/bin/env python
# -*- coding: utf-8
# Functions to analyse CSA data with nerve rootlets, vertebral levels and PMJ
# Author: Sandrine Bédard

import pandas as pd
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(
        description="TODO")
    parser.add_argument('-path-result', required=True, type=str,
                        help="Input image.")
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


def get_csa(csa_filename):
    """
    From .csv output file of process_data.sh (sct_process_segmentation),
    returns a panda dataFrame with CSA values sorted by subject eid.
    Args:
        csa_filename (str): filename of the .csv file that contains de CSA values
    Returns:
        csa (pd.Series): column of CSA values

    """
    sc_data = csv2dataFrame(csa_filename)
    csa = pd.DataFrame(sc_data[['Filename', 'MEAN(area)']]).rename(columns={'Filename': 'Subject'})
    # Add a columns with subjects eid from Filename column
    # TODO get distance PMJ, vert level or nerve level
    csa.loc[:, 'Subject'] = csa['Subject'].str.slice(-43, -32) # TODO: modifier pour ajouter ses
    # Set index to subject eid
    csa = csa.set_index('Subject')
    return csa

def 

def main():




if __name__ == '__main__':
    main()
