import cv2
import numpy as np
from pathlib import Path
from evaluate import run_evaluation

def write_pfm(output_path: Path, image: np.ndarray):
    assert image.dtype == np.float32
    assert image.ndim == 2
    h, w = image.shape
    with open(str(output_path), "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{w} {h}\n".encode())
        f.write(b"-1.0\n")  # little endian
        np.flipud(image).tofile(f)

class BaseAlgo:
    def __init__(self, dataset_dir: Path, algo_name: str, bad_thre=2.0):
        dataset_dir = dataset_dir / "MiddEval3"
        self.dataset_dir = dataset_dir
        self.train_dataset_dir = dataset_dir / "trainingF"
        self.test_dataset_dir = dataset_dir / "testF"
        self.algo_name = algo_name
        self.bad_thre = bad_thre
    
    def compute_disparity(self, left, right):
        raise NotImplementedError
    
    def run_train(self):
        for scene_dir in self.train_dataset_dir.glob("*"):
            scene_name = scene_dir.name
            print("prcessing ", scene_name)
            left_image = cv2.imread(str(scene_dir / "im0.png"), cv2.IMREAD_UNCHANGED)
            right_image = cv2.imread(str(scene_dir / "im1.png"), cv2.IMREAD_UNCHANGED)
            assert left_image.shape[-1] == 3
            assert right_image.shape[-1] == 3

            disp = self.compute_disparity(left=left_image, right=right_image)
            write_pfm(self.train_dataset_dir / f"disp0{self.algo_name}.pfm", disp)

        print("[run_train]: finished")
        print()

    def eval_train(self):
        print("Start to run evaluation!!!!!!!")
        run_evaluation(
            self.train_dataset_dir,
            algo_name=self.algo_name,
            bad_thre=self.bad_thre
        )
        print("[eval_train]: finished")
        print()
        