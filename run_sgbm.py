import cv2
import numpy as np
from pathlib import Path
from common import BaseAlgo

class SGBMAlgo(BaseAlgo):
    def __init__(self, dataset_dir, algo_name, bad_thre=2):
        super().__init__(dataset_dir, algo_name, bad_thre)
        
        self.num_disparities = 256
        self.block_size = 7

        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=24 * self.block_size * self.block_size,
            P2=96 * self.block_size * self.block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

    def compute_disparity(self, left, right):
        if left.ndim == 3:
            left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disp_raw = self.sgbm.compute(left, right).astype(np.float32)
        disp = disp_raw / 16.0

        # Middlebury 约定：无效视差用 inf
        disp[disp_raw <= 0] = np.inf

        return disp

if __name__ == "__main__":
    dataset_dir = Path("/root/workspace/middlebury-stereo-dataset/")
    algo_wrapper = SGBMAlgo(
        dataset_dir=dataset_dir,
        algo_name="SGBM",
        bad_thre=2.0,
    )
    algo_wrapper.run_train()
    algo_wrapper.eval_train(extra_algo_names=["ELAS"])
