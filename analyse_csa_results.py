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
    df2 = df[["Nerve", "Distance - PMJ (mm)", "Distance - Disc (mm)"]]

    for method in methods:
        stats[method] = {}
        for subject in subjects:
            stats[method][subject] = {}
            for metric in metrics:
                stats[method][subject][metric] = []
    for subject in subjects:
        for method in methods:
            sub1 = subject + "_ses-headNormal"
            sub2 = subject + "_ses-headDown"
            sub3 = subject + "_ses-headUp"
            subs = [sub1, sub2, sub3]
            df.loc[subs, "Subject_id"] = subject
            for level in levels:
                df3 = df2.loc[df['Nerve'] == level]
                df3 = df3.loc[df3.index.intersection(subs)]
                stats[method][subject]['mean'].append(np.mean(df3[method]))
                stats[method][subject]['std'].append(np.std(df3[method]))
            stats[method][subject]['COV'] = np.true_divide(stats[method][subject]['std'], np.abs(stats[method][subject]['mean'])).tolist()

    mean_COV_perlevel = {}
    cov = []
    for method in methods:
        mean_COV_perlevel[method] = np.zeros(6)
    for method in methods:
        statistics = list(stats[method].values())
        i = 0
        for element in statistics:
            cov += element['COV']
            i = i+1
            mean_COV_perlevel[method] = mean_COV_perlevel[method] + element['COV']
        mean_COV_perlevel[method] = mean_COV_perlevel[method]/i
    #cov = np.reshape(cov, (10, 6))
    #cov_pmj = cov[0:5, :]
    #cov_disc = cov[5:10, :]
    new_levels = np.tile(levels, 10)
    data = pd.DataFrame(columns=['Levels', 'COV', 'Method'])
    data['Levels'] = new_levels
    data['COV'] = [x * 100 for x in cov]
    data['Subject'] = np.tile(np.ravel(np.repeat(subjects, 6)),2)
    data.loc[0:30, 'Method'] = 'Distance - PMJ (mm)'
    data.loc[30:60, 'Method'] = 'Distance - Disc (mm)'
    plt.figure()
    sns.boxplot(x='Levels', y='COV', data=data, hue='Method', palette="Set3")
    plt.ylabel('COV (%)')
    plt.savefig('scatterplot_cov.png')
    plt.close()
    scatter_plot_distance(x1=data.loc[0:30, 'Levels'], y1=data.loc[0:30, 'COV'], x2=data.loc[30:60, "Levels"], y2=data.loc[30:60, 'COV'], hue=data['Subject'])
    return stats
    # Add total mean


def scatter_plot_distance(x1, y1, x2, y2, hue=None):
    plt.figure()
    fig, ax = plt.subplots(1, 2, sharey=True)
    sns.scatterplot(ax=ax[0], x=x1, y=y1, hue=hue, alpha=1, edgecolors=None, linewidth=0, palette="Spectral")
    #ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_ylabel('COV (%)')
    ax[0].set_xlabel('Level')
    ax[0].set_title('a) COV PMJ-Nerve Roots perlevel')
    ax[0].set_box_aspect(1)

    sns.scatterplot(ax=ax[1], x=x2, y=y2, hue=hue, alpha=1, edgecolors=None, linewidth=0, palette="Spectral", legend=False)
    ax[1].set_ylabel('COV (%)')
    ax[1].set_xlabel('Level')
    ax[1].set_title('b) COV Disc-Nerve Roots perlevel')
    ax[1].set_box_aspect(1)
    plt.tight_layout()
    filename = "scatterplot_distance.png"
    plt.savefig(filename)
    plt.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Read distance from nerve rootlets
    path_distance = os.path.join(args.path_results, "disc_pmj_distance.csv")
    df_distance = csv2dataFrame(path_distance).set_index('Subject')
    stats = compute_distance_mean(df_distance)

if __name__ == '__main__':
    main()
