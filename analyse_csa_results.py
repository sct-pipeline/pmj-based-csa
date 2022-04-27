#!/usr/bin/env python
# -*- coding: utf-8
# Functions to analyse CSA data with nerve rootlets, vertebral levels and PMJ
# Author: Sandrine Bédard

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
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
    csa.loc[:, 'Subject'] = csa['Subject'].str.slice(-43, -32)  # TODO: modifier pour ajouter "ses"
    # Set index to subject eid
    csa = csa.set_index('Subject')
    return csa


def compute_distance_mean(df):
    subjects = ["sub-003", "sub-004", "sub-005", "sub-006", "sub-007"]
    metrics = ['mean', 'std', 'COV']
    levels = [2, 3, 4, 5, 6, 7]
    methods = ['Distance - PMJ (mm)', 'Distance - Disc (mm)']
    stats_pmj = {}
    stats_disc = {}
    df2 = df[["Nerve", "Disc", "Distance - PMJ (mm)", "Distance - Disc (mm)"]]

    for subject in subjects:
        stats_pmj[subject] = {}
        stats_disc[subject] = {}
        for metric in metrics:
            stats_pmj[subject][metric] = []
            stats_disc[subject][metric] = []
    for subject in subjects:
        sub1 = subject + "_ses-headNormal"
        sub2 = subject + "_ses-headDown"
        sub3 = subject + "_ses-headUp"
        subs = [sub1, sub2, sub3]
        df.loc[subs, "Subject_id"] = subject
        for level in levels:
            df_pmj = df2.loc[df['Nerve'] == level]
            df_disc = df2.loc[df['Disc'] == level]
            df3_pmj = df_pmj.loc[df_pmj.index.intersection(subs)]
            df_disc = df_disc.loc[df_disc.index.intersection(subs)]
            stats_pmj[subject]['mean'].append(np.mean(df3_pmj[methods[0]]))
            stats_pmj[subject]['std'].append(np.std(df3_pmj[methods[0]]))
            stats_disc[subject]['mean'].append(np.mean(df_disc[methods[1]]))
            stats_disc[subject]['std'].append(np.std(df_disc[methods[1]]))

        stats_pmj[subject]['COV'] = np.true_divide(stats_pmj[subject]['std'], np.abs(stats_pmj[subject]['mean'])).tolist()
        stats_disc[subject]['COV'] = np.true_divide(stats_disc[subject]['std'], np.abs(stats_disc[subject]['mean'])).tolist()
    cov = []
    std = []
    mean_COV_perlevel_pmj = np.zeros(6)
    mean_COV_perlevel_disc = np.zeros(6)

    mean_std_perlevel_pmj = np.zeros(6)
    mean_std_perlevel_disc = np.zeros(6)

    statistics_pmj = list(stats_pmj.values())
    statistics_disc = list(stats_disc.values())

    i = 0
    for element_pmj in statistics_pmj:
        cov += element_pmj['COV']
        std += element_pmj['std']
        i = i+1
        mean_COV_perlevel_pmj = mean_COV_perlevel_pmj + element_pmj['COV']
        mean_std_perlevel_pmj = mean_std_perlevel_pmj + element_pmj['std']
    mean_std_perlevel_pmj = mean_std_perlevel_pmj/i
    mean_COV_perlevel_pmj = mean_COV_perlevel_pmj/i
    i = 0
    for element_disc in statistics_disc:
        cov += element_disc['COV']
        std += element_disc['std']
        i = i+1
        mean_COV_perlevel_disc = mean_COV_perlevel_disc + element_disc['COV']
        mean_std_perlevel_disc = mean_std_perlevel_disc + element_disc['std']

    mean_std_perlevel_disc = mean_std_perlevel_disc/i
    mean_COV_perlevel_disc = mean_COV_perlevel_disc/i

    new_levels = np.tile(levels, 10)
    data = pd.DataFrame(columns=['Levels', 'COV', 'Method'])
    data['Levels'] = new_levels
    data['COV'] = [x * 100 for x in cov]
    data['Subject'] = np.tile(np.ravel(np.repeat(subjects, 6)), 2)
    data.loc[0:30, 'Method'] = 'PMJ'
    data.loc[30:60, 'Method'] = 'Disc'

    plt.figure()
    plt.title('Coeeficient of variation (COV) of distance among 3 neck positions')
    sns.boxplot(x='Levels', y='COV', data=data, hue='Method', palette="Set3")
    plt.ylabel('COV (%)')
    plt.savefig('boxplot_cov.png')
    plt.close()
    scatter_plot_distance(x1=data.loc[0:30, 'Levels'],
                          y1=data.loc[0:30, 'COV'],
                          x2=data.loc[30:60, "Levels"],
                          y2=data.loc[30:60, 'COV'],
                          y_label='COV (%)',
                          title_1='a) COV PMJ-Nerve Roots perlevel',
                          title_2='b) COV Disc-Nerve Roots perlevel',
                          hue=data['Subject'],
                          filename='scatterplot_cov.png')

    # For STD
    data_std = pd.DataFrame(columns=['Levels', 'std', 'Method'])
    data_std['Levels'] = new_levels
    data_std['std'] = std
    data_std['Subject'] = np.tile(np.ravel(np.repeat(subjects, 6)),2)
    data_std.loc[0:30, 'Method'] = 'PMJ'
    data_std.loc[30:60, 'Method'] = 'Disc'
    plt.figure()
    plt.title('Standard deviation (std) of distance among 3 neck positions')
    sns.boxplot(x='Levels', y='std', data=data_std, hue='Method', palette="Set3")
    plt.ylabel('std (mm)')
    plt.savefig('boxplot_std.png')
    plt.close()
    scatter_plot_distance(x1=data_std.loc[0:30, 'Levels'],
                          y1=data_std.loc[0:30, 'std'],
                          x2=data_std.loc[30:60, "Levels"],
                          y2=data_std.loc[30:60, 'std'],
                          hue=data_std['Subject'],
                          y_label='std (mm)',
                          title_1='a) STD PMJ-Nerve Roots perlevel',
                          title_2='b) STD Disc-Nerve Roots perlevel',
                          filename='scatterplot_std.png')


def scatter_plot_distance(x1, y1, x2, y2, y_label, title_1, title_2, hue=None, filename=None):
    plt.figure()
    fig, ax = plt.subplots(1, 2, sharey=True)
    #fig, ax = plt.subplots(1, 2)

    sns.scatterplot(ax=ax[0], x=x1, y=y1, hue=hue, alpha=1, edgecolors=None, linewidth=0, palette="Spectral")
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel('Level')
    ax[0].set_title(title_1)
    ax[0].set_box_aspect(1)

    sns.scatterplot(ax=ax[1], x=x2, y=y2, hue=hue, alpha=1, edgecolors=None, linewidth=0, palette="Spectral", legend=False)
    ax[1].set_ylabel(y_label)
    ax[1].set_xlabel('Level')
    ax[1].set_title(title_2)
    ax[1].set_box_aspect(1)
    plt.tight_layout()
    if not filename:
        filename = "scatterplot.png"
    plt.savefig(filename)
    plt.close()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Read distance from nerve rootlets
    path_distance = os.path.join(args.path_results, "disc_pmj_distance.csv")
    df_distance = csv2dataFrame(path_distance).set_index('Subject')
    compute_distance_mean(df_distance)

if __name__ == '__main__':
    main()
