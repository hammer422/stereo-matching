import numpy as np
import imageio.v2 as imageio
from pathlib import Path
from collections import defaultdict

def read_pfm(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        header = f.readline().decode('latin-1').strip()
        if header not in ('PF', 'Pf'):
            raise ValueError("Not a PFM file")
        width, height = map(int, f.readline().decode('latin-1').split())
        scale = float(f.readline().decode('latin-1').strip())
        endian = '<' if scale < 0 else '>'  # negative scale => little endian
        data = np.frombuffer(f.read(), dtype=endian + 'f4')
        if header == 'Pf':  # grayscale
            img = data.reshape((height, width), order='C')
        else:  # color
            assert False
            img = data.reshape((height, width, 3), order='C')
        return np.flipud(img)  # PFM is stored flipped vertically
    
def read_calib(path):
    calib = {}
    with open(str(path)) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=')
                calib[k] = v
    return calib

def evaldisp(disp: np.ndarray, gtdisp: np.ndarray, mask: np.ndarray, 
             badthresh: float, maxdisp: int, rounddisp: bool):
    h, w = gtdisp.shape
    h2, w2 = disp.shape
    scale = w // w2

    if scale not in (1, 2, 4) or scale * w2 != w or scale * h2 != h:
        raise ValueError(
            f"GT disp size must be exactly 1, 2, or 4 * disp size\n"
            f"disp: {w2}x{h2}, GT: {w}x{h}"
        )
    
    usemask = mask is not None
    if usemask and mask.shape != gtdisp.shape:
        raise ValueError("mask image must have same size as GT")
    
    n = 0
    bad = 0 
    invalid = 0
    serr = 0.0
    for y in range(h):
        for x in range(w):
            gt = gtdisp[y,x]
            if np.isinf(gt):
                continue
            d = scale * disp[int(y / scale), int(x / scale)]
            valid = not np.isinf(d)
            if valid:
                maxd = scale * maxdisp # max disp range
                d = max(0, min(maxd, d)) # clip disps to max disp range
            if valid and rounddisp:
                d = round(d)
            err = np.fabs(d - gt)
            if usemask and mask[y,x] != 255:
                # don't evaluate pixel
                pass
            else:
                n += 1
                if valid:
                    serr += err
                    if err > badthresh:
                        bad += 1
                else:
                    invalid += 1   
    
    badpercent = 100.0 * bad / n
    invalidpercent = 100.0 * invalid / n
    totalbadpercent = 100.0 * (bad + invalid) / n
    avgErr = serr / (n - invalid)
    return {
        "coverage": 100.0 * n / (w * h),
        "badpercent": badpercent,
        "invalidpercent": invalidpercent,
        "totalbadpercent": totalbadpercent,
        "avgErr": avgErr,
    }

def print_header(bad_thre):
    print(
        f"{'dataset':14s}"
        f"{'algorithm':15s}"
        f"{'cover%':>8s}"
        f"{f'bad{bad_thre:.1f}':>8s}"
        f"{'invalid':>9s}"
        f"{'totbad':>9s}"
        f"{'avgErr':>9s}"
    )

def print_row(scene, algo, r):
    print(
        f"{scene:14s}"
        f"{algo:15s}"
        f"{r['coverage']:8.1f}"
        f"{r['badpercent']:8.2f}"
        f"{r['invalidpercent']:9.2f}"
        f"{r['totalbadpercent']:9.2f}"
        f"{r['avgErr']:9.3f}"
    )

def run_evaluation(dataset_dir: Path, algo_names, bad_thre=2.0):
    total_avg_errs = defaultdict(list)

    print_header(bad_thre)

    for scene_dir in dataset_dir.glob("*"):
        print("=" * 80)
        for algo_name in algo_names:
            calib_meta = read_calib(scene_dir / "calib.txt")
            gtdisp = read_pfm(str(scene_dir / "disp0GT.pfm"))
            disp = read_pfm(str(scene_dir / f"disp0{algo_name}.pfm"))
            mask = imageio.imread(str(scene_dir / "mask0nocc.png"))
            assert np.array(mask).ndim == 2
            assert np.array(gtdisp).ndim == 2
            assert np.array(disp).ndim == 2
            max_disp = int(calib_meta["ndisp"])
            rounddisp = int(calib_meta["isint"]) == 1

            result = evaldisp(
                disp=disp, gtdisp=gtdisp, mask=mask,
                badthresh=bad_thre,
                maxdisp=max_disp, rounddisp=rounddisp
            )
            print_row(scene_dir.name, algo_name, result)

            total_avg_errs[algo_name].append(result["avgErr"])

    print("=" * 80)
    print("total avg err: ")
    for algo_name, errs in total_avg_errs.items():
        print(f"algo_name: {algo_name},  total avg err: {np.mean(errs)}")

if __name__ == "__main__":
    dataset_dir = Path("/root/workspace/middlebury-stereo-dataset/")
    train_dataset_dir = dataset_dir / "MiddEval3" / "trainingF"
    test_dataset_dir = dataset_dir / "MiddEval3" / "testF"

    run_evaluation(
        train_dataset_dir,
        algo_names=["ELAS", "ELAS"],
        bad_thre=2.0
    )
