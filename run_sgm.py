
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from common import BaseAlgo


@dataclass
class SGMParams:
    num_paths: int # 聚合路径数
    min_disparity: int # 最小视差
    max_disparity: int # 最大视差
    p1: int # 惩罚项参数P1
    p2: int # 惩罚项参数P2


class SGMAlgoImpl:
    def __init__(self, sgm_params: SGMParams):
        self.sgm_params = sgm_params
        
    def _init(
        self,
        width, 
        height,
    ):
        pass

    def _match(
        self,
        img_left,
        img_right,
    ):
        pass

    def _reset(
        self,
        width,
        height,
    ):
        pass


class SGMAlgo(BaseAlgo):
    def __init__(self, sgm_params: SGMParams, dataset_dir, algo_name, bad_thre=2):
        super().__init__(dataset_dir, algo_name, bad_thre)
        




        