import cv2
import os
import argparse
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr


def compute_psnr(img_gt, img_gen):
    psnr = compare_psnr(img_gt, img_gen)
    return psnr


def compute_ssim(img_gt, img_gen):
    img_gt_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
    img_gen_gray = cv2.cvtColor(img_gen, cv2.COLOR_BGR2GRAY)
    (score, diff) = compare_ssim(img_gt_gray, img_gen_gray, full=True)
    return score


def path_gen(dir_in):
    message = os.walk(dir_in)
    for i, item in enumerate(message):
        filenames = item[2]
    return filenames
