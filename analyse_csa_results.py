#!/usr/bin/env python
# -*- coding: utf-8
# Functions to analyse CSA data with nerve rootlets, vertebral levels and PMJ
# Author: Sandrine Bédard

import pandas as pd
import argparse
import numpy as np
import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use('seaborn')


def get_parser():
    parser = argparse.ArgumentParser(
        description="TODO")
    parser.add_argument('-path-results', required=True, type=str,
                        help="Input image.")  # TODO : à modifier
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
    csa.loc[:, 'Subject'] = csa['Subject'].str.slice(-43, -32) # TODO: modifier pour ajouter "ses"
    # Set index to subject eid
    csa = csa.set_index('Subject')
    return csa


def compute_distance_mean(df):
    subjects = ["sub-003", "sub-004", "sub-005", "sub-006", "sub-007"]
    metrics = ['mean', 'std', 'COV']
    levels = [2, 3, 4, 5, 6, 7]
    methods = ['Distance - PMJ (mm)', 'Distance - Disc (mm)']
    stats = {}
    df2 = df[["Distance - PMJ (mm)", "Distance - Disc (mm)"]]
    
    for contrast in subjects:
        stats[contrast] = {}
        for level in levels:
            stats[contrast][level] = {}
            for method in methods:
                stats[contrast][level][method] = {}
                for metric in metrics:
                    stats[contrast][level][method][metric] = {}
    for subject in subjects:
        for level in levels:
            for method in methods:
                sub1 = subject + "_ses-headNormal"
                sub2 = subject + "_ses-headDown"
                sub3 = subject + "_ses-headUp"
                subs = [sub1, sub2, sub3]
                df.loc[subs, "Subject_id"] = subject
                stats[subject][level][method]['mean'] = np.mean(df2.loc[subs, method])
                stats[subject][level][method]['std'] = np.std(df2.loc[subs, method])
                stats[subject][level][method]['COV'] = stats[subject][level][method]['std']/stats[subject][level][method]['mean']
    print(stats)
    scatter_plot_distance(x1=df["Nerve"], y1=df["Distance - PMJ (mm)"], x2=df["Nerve"], y2=df["Distance - Disc (mm)"], hue=df["Subject_id"])
    return stats
    # Add total mean


def scatter_plot_distance(x1, y1, x2, y2, hue=None):
    plt.figure()
    fig, ax = plt.subplots(1, 2)
    sns.scatterplot(ax=ax[0], x=x1, y=y1, hue=hue, alpha=0.7, edgecolors=None, linewidth=0, palette="Spectral")
    #ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_ylabel('Distance (mm)')
    ax[0].set_xlabel('Level')
    ax[0].set_title('a) Distance PMJ-Nerve Roots perlevel')
    ax[0].set_box_aspect(1)

    sns.scatterplot(ax=ax[1], x=x2, y=y2, hue=hue, alpha=0.8, edgecolors=None, linewidth=0, palette="Spectral")
    ax[1].set_ylabel('Distance (mm)')
    ax[1].set_xlabel('Level')
    ax[1].set_title('b) Distance Disc-Nerve Roots perlevel')
    ax[1].set_box_aspect(1)
    #plt.tight_layout()
    filename = "scatterplot_distance.png"
    plt.savefig(filename)
    plt.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Read distance from nerve rootlets
    path_distance = os.path.join(args.path_results, "disc_pmj_distance.csv")
    df_distance = csv2dataFrame(path_distance).set_index('Subject')
    print(df_distance.head())
    stats = compute_distance_mean(df_distance)
    df_stats = pd.DataFrame.from_dict(stats, orient='index').transpose()
    df_stats.columns = pd.MultiIndex.from_tuples(df_stats.columns)
    print(df_stats)

if __name__ == '__main__':
    main()
