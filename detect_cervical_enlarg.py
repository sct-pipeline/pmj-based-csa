#!/usr/bin/env python
# -*- coding: utf-8
# Functions to get distance from PMJ for processing segmentation data
# Author: Sandrine Bédard

import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate graph of CSA perslice to detect cervical enlargement")
    parser.add_argument('-filename', required=True, type=str,
                        help="Input .csv file with CSA computed perslice.")
    parser.add_argument('-o', required=False, type=str,
                        default='csa.png',
                        help="Ouput graph filename.")

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
    csa = pd.DataFrame(sc_data[['Filename', 'Slice (I->S)', 'MEAN(area)']]).rename(columns={'Filename': 'Subject'})
    # Add a columns with subjects eid from Filename column
    csa.loc[:, 'Subject'] = csa['Subject'].str.slice(-43, -32)
    
    #TODO change CSA to float!!!
    
    # Set index to subject eid
    csa = csa.set_index('Subject')
    return csa

def main():
    parser = get_parser()
    args = parser.parse_args()
    df = get_csa(args.filename)
    #df[df.rebounds != 'None']
    df = df.replace('None', pd.np.nan)
    df = df.dropna(0, how='any').reset_index(drop=True)
    print(df['MEAN(area)'].to_numpy())
    plt.figure()
    plt.plot(df['Slice (I->S)'].to_numpy(), pd.to_numeric(df['MEAN(area)']).to_numpy(), 'r')
    plt.ylabel('CSA [$mm^2$]')
    plt.xlabel('Slice (I-->S)')
    plt.savefig(args.o)

if __name__ == '__main__':
    main()