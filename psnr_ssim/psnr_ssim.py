import cv2
import os
import argparse
import numpy as np
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from utils import *


# img_index = ["/is_1/rst_1", "/is_2/rst_2", "/is_3/rst_3", "/is_4/rst_4"]
img_index = ["/is"]


if __name__ == "__main__":
    dir_img_gt = "./../img/gt_all"
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="./../img/gt")
    args = parser.parse_args()
    dir_img_in = args.filepath
    ans = {}
    ans["psnr"] = []
    ans["ssim"] = []
    for i in range(len(img_index)):
        dir_img_gt_t = dir_img_gt + img_index[i]
        dir_img_in_t = dir_img_in + img_index[i]
        filenames = path_gen(dir_img_gt_t)
        psnr = []
        ssim = []
        for item in filenames:
            pth_img_gt = os.path.join(dir_img_gt_t, item)
            pth_img_in = os.path.join(dir_img_in_t, item)
            img_gt = cv2.imread(pth_img_gt)
            img_in = cv2.imread(pth_img_in)
            psnr_t = compute_psnr(img_gt, img_in)
            ssim_t = compute_ssim(img_gt, img_in)
            psnr.append(psnr_t)
            ssim.append(ssim_t)
        psnr = np.array(psnr)
        ssim = np.array(ssim)
        ans["psnr"].append(np.mean(psnr))
        ans["ssim"].append(np.mean(ssim))
    print("psnr: ", ans["psnr"])
    print("ssim: ", ans["ssim"])
