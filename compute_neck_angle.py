#!/usr/bin/env python
# -*- coding: utf-8
# Compute distance between nerve rootlets and PMJ, nerve rootlets and discs along centerline
#
# For usage, type: python compute_neck_angle.py -h

# Authors: Sandrine Bédard

import argparse
import csv
import math
from webbrowser import get
import numpy as np
import nibabel as nib
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute cobb angle beteen C2 and C6")
    parser.add_argument('-centerline', required=True, type=str,
                        help=".csv file of the centerline.")
    parser.add_argument('-disclabel', required=True, type=str,
                        help="Labels of the intervertebral discs.")
    parser.add_argument('-subject', required=True, type=str,
                        help="Subject ID")
    parser.add_argument('-ses', required=True, type=str,
                        help="ses")
    parser.add_argument('-o', required=False, type=str,
                        default='angle.csv',
                        help="Output csv filename.")

    return parser


def save_Nifti1(data, original_image, filename):
    empty_header = nib.Nifti1Header()
    image = nib.Nifti1Image(data, original_image.affine, empty_header)
    nib.save(image, filename)


def find_angle(v1, v2):
    
    unit_vector1 = v1 / np.linalg.norm(v1)
    unit_vector2 = v2 / np.linalg.norm(v2)

    dot_product = np.dot(unit_vector1, unit_vector2)

    angle = np.arccos(dot_product) #angle in radian

    angle_deg = angle*180/np.pi
    return round(angle_deg, 4)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def get_vector_from_point(p1, p2):
    v = p2 - p1
    return unit_vector(v)


def project_vector(u, n):
    # Norm of n vector
    n_norm = np.linalg.norm(n)
    proj_of_u_on_n = (np.dot(u, n)/n_norm**2)*n
    return u - proj_of_u_on_n


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Create an array with centerline coordinates
    centerline = np.genfromtxt(args.centerline, delimiter=',')
    #centerline = nib.load(args.centerline).get_fdata()
    disc_label = nib.load(args.disclabel)
    idx_2 = np.where(disc_label.get_fdata() == 2.0)[-1]
    idx_6 = np.where(disc_label.get_fdata() == 6.0)[-1]
    idx_2_corr = np.abs(centerline[2] - idx_2).argmin()  # centerline doesn't necessarly start at the index 0 if the segmentation is incomplete
    idx_6_corr = np.abs(centerline[2] - idx_6).argmin()  # centerline doesn't necessarly start at the index 0 if the segmentation is incomplete
    len = 3

    p1 = centerline[:,idx_2_corr - len]
    p2 = centerline[:,idx_2_corr + len]
    p3 = centerline[:,idx_6_corr - len]
    p4 = centerline[:,idx_6_corr + len]

    v1 = get_vector_from_point(p1=p1, p2=p2)

    print('v1', v1)
    v2 = get_vector_from_point(p1=p3, p2=p4)
    print('v1', v2)
    # Project v1 and v2 on X (R-L)
    n = np.array([1,0,0])
    v1_proj = project_vector(v1, n)
    v2_proj = project_vector(v2, n)
    print('v1_proj', v1_proj, 'v2_proj', v2_proj)
    angle = find_angle(v1, v2)
    angle_proj = find_angle(v1_proj, v2_proj)

    print('angle', angle, 'degrees')
    print('angle proj', angle_proj, 'degrees')
    subject = args.subject
    ses = args.ses
    fname_out = args.o
    if not os.path.isfile(fname_out):
        with open(fname_out, 'w') as csvfile:
            header = ['Subject', 'ses', 'Angle (degrees)']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
    with open(fname_out, 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        line = [subject, ses, angle_proj]
        spamwriter.writerow(line)
    

if __name__ == '__main__':
    main()
