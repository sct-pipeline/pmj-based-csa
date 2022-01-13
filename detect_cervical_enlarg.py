#!/usr/bin/env python
# -*- coding: utf-8
# Functions to plot CSA perslice and vertebral levels
# Need sct_process_segmentation -vert 1:10 -vertfile -perslice 1
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
    csa = pd.DataFrame(sc_data[['Filename', 'Slice (I->S)', 'VertLevel', 'MEAN(area)']]).rename(columns={'Filename': 'Subject'})
    # Add a columns with subjects eid from Filename column
    csa.loc[:, 'Subject'] = csa['Subject'].str.slice(-43, -32)
    
    #TODO change CSA to float!!!
    
    # Set index to subject eid
    csa = csa.set_index('Subject')
    return csa


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def main():
    parser = get_parser()
    args = parser.parse_args()
    df = get_csa(args.filename)
    #df[df.rebounds != 'None']
    df = df.replace('None', pd.np.nan)
    df = df.dropna(0, how='any').reset_index(drop=True)
    df = df.iloc[::-1]
    plt.figure()
    fig, ax = plt.subplots(figsize=(5,6))
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
     
    # Get slices where array changes value
    vert = df['VertLevel'].to_numpy()
    ind_vert = np.where(vert[:-1] != vert[1:])[0]
    for x in ind_vert:
        plt.axhline(x, color='k', linestyle='--')
        ax.text(x/len(ind_vert), 0.05, vert[x], transform=ax.transAxes)
    
    plt.plot(smooth(pd.to_numeric(df['MEAN(area)']).to_numpy(), 6)[1:-1],df['Slice (I->S)'].to_numpy()[1:-1], 'r', aa=True)
    plt.grid()
    plt.title('Aire transversale de la moelle épinière', fontsize=16)
    plt.ylim(max(df['Slice (I->S)'].to_numpy()[6:-6]), min(df['Slice (I->S)'].to_numpy()[6:-6]))
    plt.xlabel('ATM ($mm^2$)', fontsize=14)
    plt.ylabel('Coupe (S->I)', fontsize=14)
    #ax2.set_xlabel('VertLevel')
    plt.savefig(args.o)

if __name__ == '__main__':
    main()