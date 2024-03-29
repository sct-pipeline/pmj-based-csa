#!/usr/bin/env python
# -*- coding: utf-8
# Functions to analyse CSA data with nerve rootlets, vertebral levels and PMJ
# Author: Sandrine Bédard

from re import L
import warnings
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
from statsmodels.stats.anova import AnovaRM
import pingouin as pg

FNAME_LOG = 'log.txt'
#MY_PAL = {"PMJ": "#C1BED6", "Disc": "#96CAC1", "Spinal Roots":"#F6F6BC"}
MY_PAL = {"PMJ": "cornflowerblue", "Disc": "lightyellow", "Spinal Roots":"gold"}
# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # default: logging.DEBUG, logging.INFO
hdlr = logging.StreamHandler(sys.stdout)
logging.root.addHandler(hdlr)

plt.rcParams['axes.axisbelow'] = True


def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes statistics about CSA and distances between PMJ, dics and nerve rootlets.")
    parser.add_argument('-path-results', required=True, type=str,
                        help="Path of the results. Example: ~/pmj-csa-results/results.")
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
    csa.replace('None', np.NaN, inplace=True)
    csa = csa.astype({"MEAN(area)": float})
    csa.loc[:,'ses'] =  (csa['Subject'].str.split('/')).str[-3]
    csa.loc[:, 'Subject'] = (csa['Subject'].str.split('/')).str[-4]
    return csa


def get_RL_angle(csa_filename):
    """
    From .csv output file of process_data.sh (sct_process_segmentation),
    returns a panda dataFrame with RL angle values sorted by subject eid.
    Args:
        csa_filename (str): filename of the .csv file that contains de CSA values
    Returns:
        csa (pd.Series): column of RL angle values

    """
    sc_data = csv2dataFrame(csa_filename)
    angle = pd.DataFrame(sc_data[['Filename', 'MEAN(angle_RL)', 'Slice (I->S)']]).rename(columns={'Filename': 'Subject'})
    angle.replace('None', np.NaN, inplace=True)
    angle = angle.astype({"MEAN(angle_RL)": float})
    angle['ses'] = (angle['Subject'].str.split('/')).str[-3]
    angle.loc[:, 'Subject'] = (angle['Subject'].str.split('/')).str[-4]
    return angle


def compute_anova(df,level, depvar='std', subject='Subject', within=['Method'], aggregate_func=None):
    logger.info(level)
    logger.info('ANOVA: {}'.format(AnovaRM(data=df, depvar=depvar, subject=subject, within=within, aggregate_func=aggregate_func).fit()))


def compute_distance_mean(df):
    """"
    Analyse distance data. Compute the std of distances for each level, mean(std), generate a scatterplot.
    Args:
        df (pandas.DataFrame): Dataframe of distance .csv file.

    Returns:

    """
    # Retreive subjects ID:
    subjects = np.unique(np.array([sub.split('_')[0] for sub in df.index]))
    metrics = ['mean', 'std', 'COV']
    levels = [2, 3, 4, 5, 6, 7]
    methods = ['Distance - PMJ (mm)', 'Distance - Disc (mm)']
    stats_pmj = {}
    stats_disc = {}
    df2 = df[["Nerve", "Disc", "Distance - PMJ (mm)", "Distance - Disc (mm)"]]
    # Initialize empty dict
    for subject in subjects:
        stats_pmj[subject] = {}
        stats_disc[subject] = {}
        for metric in metrics:
            stats_pmj[subject][metric] = []
            stats_disc[subject][metric] = []
    # Loop trough subjects and level to compute meand and std of disances perlevel
    for subject in subjects:
        sub1 = subject + "_ses-headNormal"
        sub2 = subject + "_ses-headDown"
        sub3 = subject + "_ses-headUp"
        subs = [sub1, sub2, sub3]
        df2.loc[subs, "Subject_id"] = subject
        df2.loc[sub1, "ses"] = 'Normal'
        df2.loc[sub2, "ses"] = 'Down'
        df2.loc[sub3, "ses"] = 'Up'
        for level in levels:
            df_pmj = df2.loc[df['Nerve'] == level]
            df_disc = df2.loc[df['Disc'] == level]
            df3_pmj = df_pmj.loc[df_pmj.index.intersection(subs)]
            df_disc = df_disc.loc[df_disc.index.intersection(subs)]
            if len(df3_pmj[methods[0]])>1:
                stats_pmj[subject]['mean'].append(np.mean(df3_pmj[methods[0]]))
                stats_pmj[subject]['std'].append(np.std(df3_pmj[methods[0]]))
            else:
                stats_pmj[subject]['mean'].append(np.NaN)
                stats_pmj[subject]['std'].append(np.NaN)
            if len(df_disc[methods[1]])>1:
                stats_disc[subject]['mean'].append(np.mean(df_disc[methods[1]]))
                stats_disc[subject]['std'].append(np.std(df_disc[methods[1]]))
            else:
                stats_disc[subject]['mean'].append(np.NaN)
                stats_disc[subject]['std'].append(np.NaN)

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
    # Compute mean of std(distance) across subjects perlevel
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

    new_levels = np.tile(levels, 20)
    data = pd.DataFrame(columns=['Level', 'COV', 'Method'])
    data['Level'] = new_levels
    data['COV'] = [x * 100 for x in cov]
    data['Subject'] = np.tile(np.ravel(np.repeat(subjects, 6)), 2)
    data.loc[0:59, 'Method'] = 'PMJ'
    data.loc[60:119, 'Method'] = 'Disc'
    # Create boxplot of COV(distances) perlevel across subject
    plt.figure()
    plt.grid()
    plt.title('Coeeficient of variation (COV) of distance among 3 neck positions')
    sns.boxplot(x='Level', y='COV', data=data, hue='Method', palette="Set3", showmeans=True)
    plt.ylabel('COV (%)')
    plt.xlabel('Levels')
    plt.savefig('boxplot_cov.png')
    plt.close()
    scatter_plot_distance(x1=data.loc[0:59, 'Level'],
                          y1=data.loc[0:59, 'COV'],
                          x2=data.loc[60:119, "Level"],
                          y2=data.loc[60:119, 'COV'],
                          y_label='COV (%)',
                          title_1='a) COV PMJ-Nerve Roots perlevel',
                          title_2='b) COV Disc-Nerve Roots perlevel',
                          hue=data['Subject'],
                          filename='scatterplot_cov.png')

    # # Create boxplot of STD(distances) perlevel across subject
    data_std = pd.DataFrame(columns=['Level', 'std', 'Method'])
    data_std['Level'] = new_levels
    data_std['std'] = std
    data_std['Subject'] = np.tile(np.ravel(np.repeat(subjects, 6)), 2)
    data_std.loc[0:59, 'Method'] = 'PMJ'
    data_std.loc[60:119, 'Method'] = 'Disc'
    # Compute anova
    for level in levels:
        data = data_std.loc[data_std['Level']==level]
        nan_idx = (data.loc[pd.isna(data['std']), :].index).to_numpy()
        sub_nan = data.loc[nan_idx, 'Subject']
        if not sub_nan.empty:
            data.dropna(inplace=True, axis=0)
            data.drop(data.loc[data['Subject']==sub_nan.to_numpy()[0]].index, inplace=True)
        compute_anova(data, level)
    # Compute ANOVA for all levels combined
    data = data_std
    nan_idx = (data.loc[pd.isna(data['std']), :].index).to_numpy()
    sub_nan = data.loc[nan_idx, 'Subject']
    if not sub_nan.empty:
        data.dropna(inplace=True, axis=0)
        data.drop(data.loc[data['Subject']==sub_nan.to_numpy()[0]].index, inplace=True)
    compute_anova(data_std, level='all', within=['Method', 'Level'])

    # Compute Mean of std across subject perlevel for PMJ and disc distances
    std_perlevel_accross_subject_pmj = pd.DataFrame(columns=['Mean(STD)', 'STD(STD)'])
    std_perlevel_accross_subject_pmj['Mean(STD)'] = data_std.loc[0:59].groupby('Level')['std'].mean()
    std_perlevel_accross_subject_pmj['STD(STD)'] = data_std.loc[0:59].groupby('Level')['std'].std()
    logger.info('PMJ: {}'.format(std_perlevel_accross_subject_pmj))
    logger.info('Mean STD: {} ± {}'.format(std_perlevel_accross_subject_pmj['Mean(STD)'].mean(), std_perlevel_accross_subject_pmj['Mean(STD)'].std()))

    std_perlevel_accross_subject_disc = pd.DataFrame(columns=['Mean(STD)', 'STD(STD)'])
    std_perlevel_accross_subject_disc['Mean(STD)'] = data_std.loc[60:119].groupby('Level')['std'].mean()
    std_perlevel_accross_subject_disc['STD(STD)'] = data_std.loc[60:119].groupby('Level')['std'].std()
    logger.info('DISC: {}'.format(std_perlevel_accross_subject_disc))
    logger.info('Mean STD: {} ± {}'.format(std_perlevel_accross_subject_disc['Mean(STD)'].mean(), std_perlevel_accross_subject_disc['Mean(STD)'].std()))

    # Create a boxplot of STD(distance) persubject, perlevel
    plt.figure()
    plt.grid()
    plt.title('Standard deviation (std) of distance among 3 neck positions')
    sns.boxplot(x='Level', y='std', data=data_std, hue='Method', palette='Set3', showmeans=True)
    plt.ylabel('STD (mm)')
    plt.savefig('boxplot_std.png')
    plt.close()
    scatter_plot_distance(x1=data_std.loc[0:59, 'Level'],
                          y1=data_std.loc[0:59, 'std'],
                          x2=data_std.loc[60:119, "Level"],
                          y2=data_std.loc[60:119, 'std'],
                          hue=data_std['Subject'],
                          y_label='std (mm)',
                          title_1='a) STD PMJ-Nerve Roots perlevel',
                          title_2='b) STD Disc-Nerve Roots perlevel',
                          filename='scatterplot_std.png')
    return df2

def scatter_plot_distance(x1, y1, x2, y2, y_label, title_1, title_2, hue=None, filename=None):
    """
    Create a scatterplot of 2 sets of data.
    Args:
        x1 (ndaray): array of independent variable.
        y1 (ndaray): array of dependent variable.
        x2 (ndaray): array of independent variable.
        y2 (ndaray): array of dependent variable.
        y_label (str): Label name of y for both plots.
        title_1 (str): Title of plot 1.
        title_2 (str): Title of plot 2.
        hue (str): column name for color encoding.
        filename (str): filename to save the scatterplot.

    Returns:

    """
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8,5))
    ax[0].grid(color='lightgrey')
    sns.scatterplot(ax=ax[0], x=x1, y=y1, hue=hue, alpha=1, edgecolors=None, linewidth=0, palette="Spectral", legend=False)
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel('Levels')
    ax[0].set_title(title_1)
    ax[0].set_box_aspect(1)
    
    ax[1].grid(color='lightgrey')
    ax[1].legend(loc='center right') #bbox_to_anchor =(0, 1)
    sns.scatterplot(ax=ax[1], x=x2, y=y2, hue=hue, alpha=1, edgecolors=None, linewidth=0, palette="Spectral")
    ax[1].set_ylabel(y_label)
    ax[1].set_xlabel('Levels')
    ax[1].set_title(title_2)
    ax[1].set_box_aspect(1)
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Subject')
    ax[1].get_legend().remove()

    plt.tight_layout(rect=[0,0,0.86,1])
    if not filename:
        filename = "scatterplot.png"
    plt.savefig(filename)
    plt.close()


def compute_stats_csa(csa, level_type):
    """
    Compute statistics (mean, std, COV) about spinal cord CSA.
    Compute mean(COV) of all subjects perlevel.

    Args:
        csa (pandas.DataFrame): dataframe of csa for the corresponding level_type.
        level_type (str): DistancePMJ or Levels.

    Returns:

    """

    stats = pd.DataFrame(columns=['Subject', 'Mean', 'STD', 'COV', 'Level'])
    stats = stats.astype({"Mean": float, "STD": float, 'COV': float})
    stats['Subject'] = np.repeat(csa['Subject'].unique(), 6)
    levels = [2, 3, 4, 5, 6, 7]
    stats['Level'] = np.tile(levels, 10)
    # Loop through subjects
    for subject in csa['Subject'].unique():
        if level_type == 'DistancePMJ':
            levels_pmj = np.unique(csa.loc[csa.index[csa['Subject'] == subject], level_type]).tolist()
            if len(levels_pmj) > 6:  # Only use 6 levels
                diff = len(levels_pmj) - len(levels)
                levels_pmj = levels_pmj[:-diff]
        else:
            levels_pmj = levels
            
        for level in levels_pmj:
            csa_level = csa.loc[csa[level_type] == level]
            csa_level.set_index(['Subject'], inplace=True)
            index = (stats[(stats['Level'] == levels[levels_pmj.index(level)]) & (stats['Subject'] == subject)].index.tolist())
            if csa_level.loc[subject, 'MEAN(area)'].size > 1:
                stats.loc[index, 'Mean'] = np.mean(csa_level.loc[subject, 'MEAN(area)'])
                stats.loc[index, 'STD'] = np.std(csa_level.loc[subject, 'MEAN(area)'])
                stats.loc[index, 'COV'] = 100*np.divide(stats.loc[index, 'STD'], stats.loc[index, 'Mean'])
            else:
                stats.loc[index, 'Mean'] = np.NaN
                stats.loc[index, 'STD'] = np.NaN
                stats.loc[index, 'COV'] = np.NaN

    # Compute Mean and STD across subject perlevel of STD(CSA)
    std_perlevel_accross_subject = pd.DataFrame(columns=['Mean(STD)', 'STD(STD)'])
    std_perlevel_accross_subject['Mean(STD)'] = stats.groupby('Level')['STD'].mean()
    std_perlevel_accross_subject['STD(STD)'] = stats.groupby('Level')['STD'].std()

    # Compute Mean and STD acrosss suject perlevel of COV(CSA)
    cov_perlevel_accross_subject = pd.DataFrame(columns=['Mean(COV)', 'STD(COV)'])
    cov_perlevel_accross_subject['Mean(COV)'] = stats.groupby('Level')['COV'].mean()
    cov_perlevel_accross_subject['STD(COV)'] = stats.groupby('Level')['COV'].std()
    logger.info('Mean CSA perlevel:')
    logger.info(stats.groupby('Level')['Mean'].mean())
    logger.info('STD CSA perlevel:')
    logger.info(stats.groupby('Level')['Mean'].std())

    logger.info('CSA, mean COV perlevel')
    logger.info(cov_perlevel_accross_subject)
    # Compute Mean and STD acrosss suject and level of COV(CSA)
    logger.info('CSA, COV %')
    logger.info('{} ± {} %'.format(cov_perlevel_accross_subject['Mean(COV)'].mean(), cov_perlevel_accross_subject['Mean(COV)'].std()))

    return stats


def compute_anova_csa(csa_vert, csa_pmj, csa_spinal):
    """
    Compute ANOVA fro repeated mesures on CSA results (MEAN(area)) for the following categories: 
        - Methods{PMJ, vert, spinal} 
        - Levels{2,3,4,5} --> Unbalanced data for levels 6 and 7
        - ses{Up, Normal, Down}
    Args:
        csa_vert (pandas.DataFrame): Dataframe of vertebral-based CSA.
        csa_spinal (pandas.DataFrame): Dataframe of spinal-based CSA.
        csa_pmj (pandas.DataFrame): Dataframe of pmj-based CSA.
    Returns:

    """
    csa_pmj['Level'] = np.NaN
    subjects = np.unique(np.array([sub.split('_')[0] for sub in csa_pmj['Subject']]))
    for sub in subjects:
        for ses in ['ses-headUp', 'ses-headDown', 'ses-headNormal']:
            length  = csa_pmj.loc[(csa_pmj['ses']==ses)&(csa_pmj['Subject']==sub), 'Level'].size
            csa_pmj.loc[(csa_pmj['ses']==ses)&(csa_pmj['Subject']==sub), 'Level'] = np.arange(2,length+2)
    csa_joinded = (csa_pmj.append(csa_vert, ignore_index=True)).append(csa_spinal, ignore_index=True)
    len_pmj = csa_pmj['Subject'].size
    len_vert = csa_vert['Subject'].size
    len_spinal = csa_spinal['Subject'].size
    csa_joinded.loc[0:len_pmj -1, 'Method'] = 'PMJ'
    csa_joinded.loc[len_pmj:len_pmj + len_vert - 1, 'Method'] = 'Disc'
    csa_joinded.loc[len_pmj + len_vert:len_pmj + len_vert + len_spinal -1, 'Method'] = 'Spinal Roots'
    # Drop unwanted columns for ANOVA
    csa_joinded.drop(columns=['Slice (I->S)', 'DistancePMJ'], inplace=True)
    # Remove level 1 since only for discs
    csa_joinded = csa_joinded.drop(csa_joinded.loc[(csa_joinded['Level']==1)].index)
    # Remove levels higher than 5 because of unbalanced data
    for level in np.unique(csa_joinded.loc[:, 'Level']).tolist():
        if level > 5:
            csa_joinded = csa_joinded.drop(csa_joinded.loc[(csa_joinded['Level']==level)].index)
    compute_anova(csa_joinded, level='2-3-4-5', depvar='MEAN(area)', subject='Subject', within=['Method', 'Level', 'ses'])

    # Compute only on level 3
    csa_joinded_3 = csa_joinded.loc[(csa_joinded['Level']==3)]
    compute_anova(csa_joinded_3, level='3', depvar='MEAN(area)', subject='Subject', within=['Method', 'Level', 'ses'])

    # Check only SPINAL vs PMJ
    csa_joinded_pmj = csa_joinded.drop(csa_joinded.loc[csa_joinded['Method']=='Disc'].index)
    compute_anova(csa_joinded_pmj, level='PMJ vs SPINAL', depvar='MEAN(area)', subject='Subject', within=['Method', 'Level', 'ses'])

    # Check only SPINAL vs VERT
    csa_joinded_disc = csa_joinded.drop(csa_joinded.loc[csa_joinded['Method']=='PMJ'].index)
    compute_anova(csa_joinded_disc, level='DISC vs SPINAL', depvar='MEAN(area)', subject='Subject', within=['Method', 'Level', 'ses'])

    # Check only PMJ vs VERT
    csa_joinded_disc_pmj = csa_joinded.drop(csa_joinded.loc[csa_joinded['Method']=='Spinal Roots'].index)
    compute_anova(csa_joinded_disc_pmj, level='DISC vs PMJ', depvar='MEAN(area)', subject='Subject', within=['Method', 'Level', 'ses'])

    # Check only SPINAL vs PMJ for 3
    csa_joinded_pmj3 = csa_joinded_3.drop(csa_joinded_3.loc[csa_joinded_3['Method']=='Disc'].index)
    compute_anova(csa_joinded_pmj3, level='PMJ vs SPINAL 3', depvar='MEAN(area)', subject='Subject', within=['Method', 'Level', 'ses'])

    # Check only SPINAL vs VERT for 3
    csa_joinded_disc3 = csa_joinded_3.drop(csa_joinded_3.loc[csa_joinded_3['Method']=='PMJ'].index)
    compute_anova(csa_joinded_disc3, level='DISC vs SPINAL 3', depvar='MEAN(area)', subject='Subject', within=['Method', 'Level', 'ses'])

    # Check only PMJ vs VERT for 3
    csa_joinded_disc_pmj3 = csa_joinded_3.drop(csa_joinded_3.loc[csa_joinded_3['Method']=='Spinal Roots'].index)
    compute_anova(csa_joinded_disc_pmj3, level='DISC vs PMJ 3', depvar='MEAN(area)', subject='Subject', within=['Method', 'Level', 'ses'])




def analyse_csa(csa_vert, csa_spinal, csa_pmj):
    """
    For CSA of 3 references: spinal rootlets, discs and PMJ, generate a scatterplot and boxplot across subjects perlevels.
    Args:
        csa_vert (pandas.DataFrame): Dataframe of statistics of vertebral-based CSA.
        csa_spinal (pandas.DataFrame): Dataframe of statistics of spinal-based CSA.
        csa_pmj (pandas.DataFrame): Dataframe of statistics of pmj-based CSA.

    Returns:

    """
    plt.figure()
    plt.grid()
    csa = (csa_pmj.append(csa_vert, ignore_index=True)).append(csa_spinal, ignore_index=True)
    csa.loc[0:59, 'Method'] = 'PMJ' # TODO change according to number of subject --> maybe automate
    csa.loc[60:119, 'Method'] = 'Disc'
    csa.loc[120:179, 'Method'] = 'Spinal Roots'
    plt.title('Coeficient of variation (COV) of CSA among 3 neck positions')
    sns.boxplot(x='Level', y='COV', data=csa, hue='Method', palette='Set3', showmeans=True)
    plt.ylabel('COV (%)')
    plt.savefig('boxplot_csa_COV.png')
    plt.close()

    # Scatter Plot of COV of CSA permethods and perlevel
    fig, ax = plt.subplots(1, 3, sharey=True, figsize=(10, 30))
    y_label = 'COV (%)'
    sns.scatterplot(ax=ax[0], x=csa.loc[0:59, 'Level'], y=csa.loc[0:59, 'COV'], hue=csa.loc[0:59, 'Subject'], alpha=1, edgecolors=None, linewidth=0, palette="Spectral")
    ax[0].set_ylabel(y_label)
    ax[0].set_xlabel('Levels')
    ax[0].set_title('a) COV of CSA with Disc')
    ax[0].set_box_aspect(1)
    ax[0].grid(color='lightgray')
    ax[0].set_axisbelow(True)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', title='Subject')
    ax[0].get_legend().remove()

    sns.scatterplot(ax=ax[1], x=np.array(csa.loc[60:119, 'Level']), y=csa.loc[60:119, 'COV'], hue=csa.loc[60:119, 'Subject'], alpha=1, edgecolors=None, linewidth=0, palette="Spectral", legend=False)
    ax[1].set_ylabel(y_label)
    ax[1].set_xlabel('Levels')
    ax[1].set_title('b) COV of CSA with Spinal Roots')
    ax[1].set_box_aspect(1)
    ax[1].grid(color='lightgray')
    ax[1].set_axisbelow(True)

    sns.scatterplot(ax=ax[2], x=csa.loc[120:179, 'Level'], y=csa.loc[120:179, 'COV'], hue=csa.loc[120:179, 'Subject'], alpha=1, edgecolors=None, linewidth=0, palette="Spectral", legend=False)
    ax[2].set_ylabel(y_label)
    ax[2].set_xlabel('Levels')
    ax[2].set_title('b) COV of CSA with PMJ')
    ax[2].set_box_aspect(1)
    ax[2].grid(color='lightgray')
    ax[2].set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    filename = "scatterplot_CSA_COV.png"
    plt.savefig(filename)
    plt.close()


def get_csa_files(path_results):
    """
    Get all .csv file of CSA for pmj, vert and spinal.

    Args:
        path_results (str): path of the results where are all teh .csv files.

    Returns:
        path_csa_vert (list): list of all paths to vert csa.
        path_csa_spinal (list): list of all paths to spinal csa.
        path_csa_pmj (list): list of all paths to pmj csa.
    """
    path_csa_vert = glob.glob(os.path.join(path_results, "*_csa-SC_vert.csv"), recursive=True)
    path_csa_spinal = glob.glob(os.path.join(path_results, "*_csa-SC_spinal.csv"), recursive=True)
    path_csa_pmj = glob.glob(os.path.join(path_results, "*_csa-SC_pmj.csv"), recursive=True)

    return path_csa_vert, path_csa_spinal, path_csa_pmj


def compute_neck_angle(angles):
    angles.sort_index(axis=1, inplace=True)
    ses_Up = angles.loc[angles['ses']=='ses-headUp']
    ses_Normal = angles.loc[angles['ses']=='ses-headNormal']
    ses_Down = angles.loc[angles['ses']=='ses-headDown']
    # Compute angle between C2 and C5
    angles_Up = ses_Up.loc[ses_Up['Level']==5.0]['MEAN(angle_RL)'].to_numpy() - ses_Up.loc[ses_Up['Level']==2.0]['MEAN(angle_RL)'].to_numpy() 
    angle_Normal = ses_Normal.loc[ses_Normal['Level']==5.0]['MEAN(angle_RL)'].to_numpy() - ses_Normal.loc[ses_Normal['Level']==2.0]['MEAN(angle_RL)'].to_numpy()
    angle_Down = ses_Down.loc[ses_Down['Level']==5.0]['MEAN(angle_RL)'].to_numpy() - ses_Down.loc[ses_Down['Level']==2.0]['MEAN(angle_RL)'].to_numpy()
    
    angles = pd.DataFrame()
    angles['ses'] = np.append(np.append(np.tile('Up',10), np.tile('Normal',10)), (np.tile('Down', 10)))
    
    angles['subject'] = np.tile(['sub-002', 'sub-003', 'sub-004', 'sub-005', 'sub-006', 'sub-007','sub-008', 'sub-009', 'sub-010', 'sub-011'], 3)
    angles['angle'] = np.append(np.append(angles_Up, angle_Normal),angle_Down)
    plt.figure()
    sns.scatterplot(x=angles['subject'], y=angles['angle'], hue=angles['ses'], alpha=1, edgecolors=None, linewidth=0, palette="Spectral")
    plt.savefig('angles.png')
    plt.close()
    logger.info('angles Up: \n {}'.format(angles_Up))
    logger.info('angles Normal:\n  {}'.format(angle_Normal))
    logger.info('angles Down\n  {}'.format(angle_Down))

    # Compute max angle variation between up and down
    angles_diff = angles_Up - angle_Down
    return angles_diff


def df_to_csv(df, filename):
    """
    Save a Dataframe as a .csv file.
    Args:
        df (panda.DataFrame)
        filename (str): Name of the output .csv file.
    """
    df.to_csv(filename)
    logger.info('Created: ' + filename)


def compute_corr_angles(angles, distance):
    """
    Compute correlation between difference in neck angle and the distance between nerve rootlet-Disc/PMJ between flexion and extension positions.
    Args:
        angles (ndarray): array of angle difference bewteen neck flexion and extension 
        distance (panda.DataFrame): array of distance bewteen nerve rootlet & PMJ or discs for each level
    """
    subjects = np.unique(np.array([sub.split('_')[0] for sub in distance.index]))
    df_pmj = pd.DataFrame(columns=['subject', 'angles', 'distance_pmj', 'level'])
    df_disc = pd.DataFrame(columns=['subject', 'angles', 'distance_disc', 'level'])
    df_pmj['subject'] = np.tile(subjects,6)
    df_pmj['level'] = np.tile(range(2,8),10)
    df_pmj['angles'] = np.tile(angles,6)


    df_disc['subject'] = np.tile(subjects,6)
    df_disc['level'] = np.tile(range(2,8),10)
    df_disc['angles'] = np.tile(angles,6)

    for level in range(2,8):
        for sub in subjects:
            dist_pmj_up = distance.loc[(distance['ses']=='Up')& (distance['Nerve']==level) & (distance['Subject_id']==sub),'Distance - PMJ (mm)']
            dist_pmj_up.dropna(inplace=True)
            dist_pmj_down = distance.loc[(distance['ses']=='Down')& (distance['Nerve']==level) & (distance['Subject_id']==sub),'Distance - PMJ (mm)']
            dist_pmj_down.dropna(inplace=True)
            diff_pmj = np.array(dist_pmj_up) - np.array(dist_pmj_down)

            dist_disc_up = distance.loc[(distance['ses']=='Up')& (distance['Disc']==level) & (distance['Subject_id']==sub),'Distance - Disc (mm)']
            dist_disc_up.dropna(inplace=True)
            dist_disc_down = distance.loc[(distance['ses']=='Down')& (distance['Disc']==level) & (distance['Subject_id']==sub),'Distance - Disc (mm)']
            dist_disc_down.dropna(inplace=True)
            diff_disc = np.array(dist_disc_up) - np.array(dist_disc_down)
            if bool(diff_pmj):
                df_pmj.loc[(df_pmj['level']==level) & (df_pmj['subject']==sub),'distance_pmj'] = diff_pmj[0]
            if bool(diff_disc):
                df_disc.loc[(df_disc['level']==level) & (df_disc['subject']==sub),'distance_disc'] = diff_disc[0]
    
    df_pmj = df_pmj.astype({"distance_pmj": float})
    df_pmj.dropna(inplace=True)
    df_pmj.set_index(['subject'], inplace=True)
    corr_table_pmj = df_pmj.corr(method='pearson')
    logger.info(corr_table_pmj)
    # Save a.csv file of the correlation matrix in the results folder
    corr_filename = 'corr_angle_distance_pmj'
    df_to_csv(corr_table_pmj, corr_filename + '.csv')

    df_disc = df_disc.astype({"distance_disc": float})
    df_disc.dropna(inplace=True)
    df_disc.set_index(['subject'], inplace=True)
    corr_table_disc = df_disc.corr(method='pearson')
    logger.info(corr_table_disc)

    # Save a.csv file of the correlation matrix in the results folder
    corr_filename = 'corr_angle_distance_disc'
    df_to_csv(corr_table_disc, corr_filename + '.csv')

    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8,5))
    ax[0].grid(color='lightgrey')
    sns.scatterplot(ax=ax[0], x=df_pmj['angles'], y=df_pmj['distance_pmj'], alpha=1, edgecolors=None, linewidth=0, palette="Spectral", legend=False)

    ax[0].set_ylabel('PMJ distance')
    ax[0].set_xlabel('Angle')
    ax[0].set_title('a) PMJ')
    ax[0].set_box_aspect(1)
    
    ax[1].grid(color='lightgrey')
    #ax[1].legend(loc='center right') #bbox_to_anchor =(0, 1)
    sns.scatterplot(ax=ax[1], x=df_disc['angles'], y=df_disc['distance_disc'], alpha=1, edgecolors=None, linewidth=0, palette="Spectral")
    ax[1].set_ylabel('Disc distance')
    ax[1].set_xlabel('Angle')
    ax[1].set_title('b) DISC')
    ax[1].set_box_aspect(1)
    #handles, labels = ax[1].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='center right', title='Subject')
    #ax[1].get_legend().remove()

    plt.tight_layout(rect=[0,0,0.86,1])
    filename = "scatterplot_angles_distance.png"
    plt.savefig(filename)
    plt.close()
    for lev in range(2,8):
        logger.info('PMJ {} \n {}'.format(lev, (df_pmj.loc[df_pmj['level']==lev]).corr(method='pearson')))
        logger.info('Disc {} \n {}'.format(lev, (df_disc.loc[df_disc['level']==lev]).corr(method='pearson')))


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
    distances = compute_distance_mean(df_distance)

    # Get.csv filenames of CSA.
    files_vert, files_spinal, files_pmj = get_csa_files(args.path_results)

    # Initialize dataframes to get CSA of 3 references.
    csa_vert = pd.DataFrame()
    csa_spinal = pd.DataFrame()
    csa_pmj = pd.DataFrame()
    angles_df = pd.DataFrame()

    # Loop through files to get all CSA values and put them in a dataframe
    # Retreive Vertebral-based CSA
    for files in files_vert:
        data = get_csa(files)
        angles = get_RL_angle(files)
        basename = os.path.basename(files)
        # Retrieve .csv files of labels correspondance.
        path_labels_corr = glob.glob(os.path.join(args.path_results, basename[0:18] + "*_vert.nii.gz_labels.csv"), recursive=True)
        labels_corr = csv2dataFrame(path_labels_corr[0])
        # Get labels correspondance for vert
        for label in data['Slice (I->S)']:
            angles.loc[angles['Slice (I->S)'] == label, 'Level'] = labels_corr.loc[labels_corr['Slices'] == label, 'Level']
            data.loc[data['Slice (I->S)'] == label, 'Level'] = labels_corr.loc[labels_corr['Slices'] == label, 'Level']
        csa_vert = csa_vert.append(data, ignore_index=True)
        angles_df = angles_df.append(angles, ignore_index=True)
    # Compute RL angle of spinal cord between disc C1-C2 and C5-C6 
    angles_diff = compute_neck_angle(angles_df)
    # Compute correlation bewteen angle and distances
    compute_corr_angles(angles_diff, distances)

    # Retreive spinal-based CSA
    for files in files_spinal:
        data = get_csa(files)
        basename = os.path.basename(files)
        # Retrve .csv files of labels correspondance.
        path = os.path.join(args.path_results, basename[0:18] + "*_discs.nii.gz_labels.csv")
        path_labels_corr = glob.glob(path, recursive=True)
        labels_corr = csv2dataFrame(path_labels_corr[0])
        # Get labels correspondance for spinal
        for label in data['Slice (I->S)']:
            data.loc[data['Slice (I->S)'] == label, 'Level'] = labels_corr.loc[labels_corr['Slices'] == label, 'Level']
        csa_spinal = csa_spinal.append(data, ignore_index=True)
    
    # Retreive PMJ-based CSA
    for files in files_pmj:
        data = get_csa(files)
        csa_pmj = csa_pmj.append(data, ignore_index=True)
    
    
    # Compute statistics about CSA.
    logger.info('Vertebral-based CSA')
    stats_csa_vert = compute_stats_csa(csa_vert, 'Level')
    logger.info('Spinal-based CSA')
    stats_csa_spinal = compute_stats_csa(csa_spinal, 'Level')
    logger.info('PMJ-based CSA')
    stats_csa_pmj = compute_stats_csa(csa_pmj, 'DistancePMJ')
    compute_anova_csa(csa_vert, csa_pmj, csa_spinal)
    analyse_csa(stats_csa_vert, stats_csa_spinal, stats_csa_pmj)


if __name__ == '__main__':
    main()
