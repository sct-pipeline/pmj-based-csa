#!/usr/bin/env python
# -*- coding: utf-8
# Functions to analyse CSA data with nerve rootlets, vertebral levels and PMJ
# Author: Sandrine Bédard

import warnings

from sklearn.metrics import label_ranking_average_precision_score
warnings.filterwarnings("ignore")
import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import logging

FNAME_LOG= 'log.txt'

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)


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
    csa = pd.DataFrame(sc_data[['Filename', 'MEAN(area)', 'Slice (I->S)', 'DistancePMJ']]).rename(columns={'Filename': 'Subject'})
    csa.replace('None',np.NaN, inplace=True)
    csa = csa.astype({"MEAN(area)": float})
    csa.loc[:, 'Subject'] = (csa['Subject'].str.split('/')).str[-4]
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
    data.loc[0:29, 'Method'] = 'PMJ'
    data.loc[30:60, 'Method'] = 'Disc'
    plt.figure()
    plt.grid()
    plt.title('Coeeficient of variation (COV) of distance among 3 neck positions')
    sns.boxplot(x='Levels', y='COV', data=data, hue='Method', palette="Set3", showmeans=True)
    plt.ylabel('COV (%)')
    plt.savefig('boxplot_cov.png')
    plt.close()
    scatter_plot_distance(x1=data.loc[0:29, 'Levels'],
                          y1=data.loc[0:29, 'COV'],
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

    # Compute Mean of std across subject perlevel
    std_perlevel_accross_subject_pmj = pd.DataFrame(columns=['Mean(STD)', 'STD(STD)'])
    std_perlevel_accross_subject_pmj['Mean(STD)'] = data_std.loc[0:29].groupby('Levels')['std'].mean()
    std_perlevel_accross_subject_pmj['STD(STD)'] = data_std.loc[0:29].groupby('Levels')['std'].std()
    logger.info('PMJ: {}'.format(std_perlevel_accross_subject_pmj))
    logger.info('Mean STD: {} ± {}'.format(std_perlevel_accross_subject_pmj['Mean(STD)'].mean(), std_perlevel_accross_subject_pmj['Mean(STD)'].std()))

    std_perlevel_accross_subject_disc = pd.DataFrame(columns=['Mean(STD)', 'STD(STD)'])
    std_perlevel_accross_subject_disc['Mean(STD)'] = data_std.loc[30:60].groupby('Levels')['std'].mean()
    std_perlevel_accross_subject_disc['STD(STD)'] = data_std.loc[30:60].groupby('Levels')['std'].std()
    logger.info('DISC: {}'.format(std_perlevel_accross_subject_disc))
    logger.info('Mean STD: {} ± {}'.format(std_perlevel_accross_subject_disc['Mean(STD)'].mean(), std_perlevel_accross_subject_disc['Mean(STD)'].std()))

    plt.figure()
    plt.grid()
    plt.title('Standard deviation (std) of distance among 3 neck positions')
    sns.boxplot(x='Levels', y='std', data=data_std, hue='Method', palette="Set3", showmeans=True)
    plt.ylabel('std (mm)')
    plt.savefig('boxplot_std.png')
    plt.close()
    scatter_plot_distance(x1=data_std.loc[0:29, 'Levels'],
                          y1=data_std.loc[0:29, 'std'],
                          x2=data_std.loc[30:60, "Levels"],
                          y2=data_std.loc[30:60, 'std'],
                          hue=data_std['Subject'],
                          y_label='std (mm)',
                          title_1='a) STD PMJ-Nerve Roots perlevel',
                          title_2='b) STD Disc-Nerve Roots perlevel',
                          filename='scatterplot_std.png')


def scatter_plot_distance(x1, y1, x2, y2, y_label, title_1, title_2, hue=None, filename=None):
    plt.grid()
    fig, ax = plt.subplots(1, 2, sharey=True)
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


def compute_stats_csa(csa, level_type):
    csa.drop(csa.index[csa['Subject'] == 'sub-002'], inplace=True)
    if level_type == 'Levels':
        levels = [2, 3, 4, 5, 6, 7]
    elif level_type == 'DistancePMJ':
        levels = [50, 60, 70, 80, 90, 100]
    stats = pd.DataFrame(columns=['Subject', 'Mean', 'STD', 'COV', 'Level'])
    stats = stats.astype({"Mean": float, "STD": float, 'COV': float})
    stats['Subject'] = np.repeat(csa['Subject'].unique(), 6)
    stats['Level'] = np.tile(levels, 5)
    for subject in csa['Subject'].unique():
        for level in levels:
            csa_level = csa.loc[csa[level_type] == level]
            csa_level.set_index(['Subject'], inplace=True)
            index = (stats[(stats['Level'] == level) & (stats['Subject'] == subject)].index.tolist())
            stats.loc[index, 'Mean'] = np.mean(csa_level.loc[subject, 'MEAN(area)'])
            stats.loc[index, 'STD'] = np.std(csa_level.loc[subject, 'MEAN(area)'])
            stats.loc[index, 'COV'] = 100*np.divide(stats.loc[index, 'STD'], stats.loc[index, 'Mean'])
    # Compute Mean and STD across subject perlevel of STD(CSA)
    std_perlevel_accross_subject = pd.DataFrame(columns=['Mean(STD)', 'STD(STD)'])
    std_perlevel_accross_subject['Mean(STD)'] = stats.groupby('Level')['STD'].mean()
    std_perlevel_accross_subject['STD(STD)'] = stats.groupby('Level')['STD'].std()

    # Compute Mean and STD acrosss suject perlevel of COV(CSA)
    cov_perlevel_accross_subject = pd.DataFrame(columns=['Mean(COV)', 'STD(COV)'])
    cov_perlevel_accross_subject['Mean(COV)'] = stats.groupby('Level')['COV'].mean()
    cov_perlevel_accross_subject['STD(COV)'] = stats.groupby('Level')['COV'].std()
    logger.info('CSA, mean COV perlevel')
    logger.info(cov_perlevel_accross_subject)
    # Compute Mean and STD acrosss suject and level of COV(CSA)
    logger.info('CSA, COV %')
    logger.info('{} ± {} %'.format(cov_perlevel_accross_subject['Mean(COV)'].mean(), cov_perlevel_accross_subject['Mean(COV)'].std()))

    return stats


def analyse_csa(csa_vert, csa_spinal, csa_pmj):
    plt.figure()
    plt.grid()
    csa = (csa_vert.append(csa_spinal, ignore_index=True)).append(csa_pmj, ignore_index=True)
    csa.loc[0:29, 'Method'] = 'Disc'
    csa.loc[30:59, 'Method'] = 'Spinal Roots'
    csa.loc[60:90, 'Method'] = 'PMJ'
    plt.title('Coeficient of variation (COV) of CSA among 3 neck positions')
    sns.boxplot(x='Level', y='COV', data=csa, hue='Method', palette="Set3", showmeans=True)
    plt.ylabel('COV (%)')
    plt.savefig('boxplot_csa_COV.png')
    plt.close()

# Scatter Plot of COV of CSA permethods and perlevel
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10,30))
    y_label = 'COV (%)'
    sns.scatterplot(ax=ax[0], x=csa.loc[0:29, 'Level'], y=csa.loc[0:29, 'COV'], hue=csa.loc[0:29, 'Subject'], alpha=1, edgecolors=None, linewidth=0, palette="Spectral")
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel('Level')
    ax[0].set_title('a) COV of CSA with Disc')
    ax[0].set_box_aspect(1)
    ax[0].grid(color='lightgray')
    ax[0].set_axisbelow(True)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Subject')
    ax[0].get_legend().remove()

    sns.scatterplot(ax=ax[1], x=np.array(csa.loc[30:59, 'Level']), y=csa.loc[30:59, 'COV'], hue=csa.loc[30:59,'Subject'], alpha=1, edgecolors=None, linewidth=0, palette="Spectral", legend=False)
    ax[1].set_ylabel(y_label)
    ax[1].set_xlabel('Level')
    ax[1].set_title('b) COV of CSA with Spinal Roots')
    ax[1].set_box_aspect(1)
    ax[1].grid(color='lightgray')
    ax[1].set_axisbelow(True)

    sns.scatterplot(ax=ax[2], x=csa.loc[60:90, 'Level'], y=csa.loc[60:90, 'COV'], hue=csa.loc[60:90,'Subject'], alpha=1, edgecolors=None, linewidth=0, palette="Spectral", legend=False)
    ax[2].set_ylabel(y_label)
    ax[2].set_xlabel('Level')
    ax[2].set_title('b) COV of CSA with PMJ')
    ax[2].set_box_aspect(1)
    ax[2].grid(color='lightgray')
    ax[2].set_axisbelow(True)
    #plt.legend(loc=(1.04,0))
    plt.tight_layout(rect=[0,0,0.88,1])
    filename = "scatterplot_CSA_COV.png"
    plt.savefig(filename)
    plt.close()


def get_csa_files(path_results):
    path_csa_vert = glob.glob(path_results + "*_vert.csv", recursive=True)
    path_csa_vert_labels = glob.glob(path_results + "*_vert.nii.gz_labels.csv", recursive=True)
    path_csa_spinal = glob.glob(path_results + "*_spinal.csv", recursive=True)
    path_csa_spinal_labels = glob.glob(path_results + "*_discs.nii.gz_labels.csv", recursive=True)    
    path_csa_pmj = glob.glob(path_results + "*_pmj.csv", recursive=True)

    return path_csa_vert, path_csa_spinal, path_csa_pmj, path_csa_vert_labels, path_csa_spinal_labels


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Dump log file there
    if os.path.exists(FNAME_LOG):
        os.remove(FNAME_LOG)
    fh = logging.FileHandler(os.path.join(os.path.abspath(os.curdir), FNAME_LOG))
    logging.root.addHandler(fh)

    # Read distance from nerve rootlets
    path_distance = os.path.join(args.path_results, "disc_pmj_distance.csv")
    df_distance = csv2dataFrame(path_distance).set_index('Subject')
    compute_distance_mean(df_distance)

    files_vert, files_spinal, files_pmj, path_csa_vert_labels, path_csa_spinal_labels = get_csa_files(args.path_results)

# TODO: put in a loop

    csa_vert = pd.DataFrame()
    csa_spinal = pd.DataFrame()
    csa_pmj = pd.DataFrame()

    for files in files_vert:
        data = get_csa(files)
        basename = os.path.basename(files)
        path_labels_corr = glob.glob(args.path_results + basename[0:18]+"*_vert.nii.gz_labels.csv", recursive=True)
        labels_corr = csv2dataFrame(path_labels_corr[0])
        # Get labels correspondance for vert
        for label in data['Slice (I->S)']:
            data.loc[data['Slice (I->S)']==label, 'Levels'] = labels_corr.loc[labels_corr['Slices']==label, 'Level']
        csa_vert = csa_vert.append(data, ignore_index=True)

    for files in files_spinal:
        data = get_csa(files)
        basename = os.path.basename(files)
        path_labels_corr = glob.glob(args.path_results + basename[0:18]+"*_discs.nii.gz_labels.csv", recursive=True)
        labels_corr = csv2dataFrame(path_labels_corr[0])
        # Get labels correspondance for spinal
        for label in data['Slice (I->S)']:
            data.loc[data['Slice (I->S)']==label, 'Levels'] = labels_corr.loc[labels_corr['Slices']==label, 'Level']
        csa_spinal = csa_spinal.append(data, ignore_index=True)

    for files in files_pmj:
        data = get_csa(files)
        csa_pmj = csa_pmj.append(data, ignore_index=True)
    stats_csa_vert = compute_stats_csa(csa_vert, 'Levels')
    stats_csa_spinal = compute_stats_csa(csa_spinal, 'Levels')
    stats_csa_pmj = compute_stats_csa(csa_pmj, 'DistancePMJ')
    analyse_csa(stats_csa_vert, stats_csa_spinal, stats_csa_pmj)

if __name__ == '__main__':
    main()
