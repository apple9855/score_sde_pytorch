#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import csv
import subprocess
import numpy as np
from PIL import Image

# ----------------------------
# Paths aligned with 03_run_sampling.sh
# ----------------------------
OUT_DIR = "engineering_validation/results"
ART_BASE = "engineering_validation/artifacts"

# IMPORTANT: must match DEST="${ART_DIR}/${NAME}" in 03.sh
VE_RUN = "VE_ncsnpp_ckpt24"
VP_RUN = "VP_ddpmpp_ckpt26"

VE_DIR = os.path.join(ART_BASE, VE_RUN)
VP_DIR = os.path.join(ART_BASE, VP_RUN)

CSV_PATH  = os.path.join(OUT_DIR, "summary.csv")
REPORT_MD = os.path.join(OUT_DIR, "REPORT.md")
ENV_TXT   = os.path.join(OUT_DIR, "RUN_ENV.txt")

IMG_VE   = os.path.join(OUT_DIR, "ve_samples_grid.png")
IMG_VP   = os.path.join(OUT_DIR, "vp_samples_grid.png")
IMG_SIDE = os.path.join(OUT_DIR, "ve_vs_vp_side_by_side.png")


def cmd_out(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"<unavailable: {e}>"


def pick_one(pattern, prefer_basename=None):
    cands = sorted(glob.glob(pattern, recursive=True))
    if not cands:
        return None
    if prefer_basename:
        for p in cands:
            if os.path.basename(p) == prefer_basename:
                return p
    return cands[0]


def load_report_metrics(report_path):
    z = np.load(report_path)
    IS  = float(z["IS"])  if "IS"  in z.files else None
    fid = float(z["fid"]) if "fid" in z.files else None
    kid = float(z["kid"]) if "kid" in z.files else None
    return IS, fid, kid


def load_samples(samples_path):
    z = np.load(samples_path)
    samples = z["samples"]

    # Defensive: ensure uint8
    if samples.dtype != np.uint8:
        samples = np.clip(samples, 0, 255).astype(np.uint8)

    # Defensive: accept either NHWC or NCHW
    if samples.ndim != 4:
        raise ValueError(f"Unexpected samples shape: {samples.shape} (expected 4D)")

    # If looks like NCHW (common DL format): convert to NHWC
    # Heuristic: channel dim is 1 or 3
    if samples.shape[1] in (1, 3) and samples.shape[-1] not in (1, 3):
        samples = np.transpose(samples, (0, 2, 3, 1))

    return samples


def save_grid(samples_path, out_png, n=64, cols=8):
    s = load_samples(samples_path)
    n = min(n, s.shape[0])
    rows = (n + cols - 1) // cols
    H, W = s.shape[1], s.shape[2]

    grid = Image.new("RGB", (cols * W, rows * H), (255, 255, 255))
    for i in range(n):
        r, c = divmod(i, cols)
        grid.paste(Image.fromarray(s[i]), (c * W, r * H))
    grid.save(out_png)


def side_by_side(left_png, right_png, out_png):
    L = Image.open(left_png).convert("RGB")
    R = Image.open(right_png).convert("RGB")

    h = max(L.height, R.height)
    if L.height != h:
        L = L.resize((int(L.width * (h / L.height)), h))
    if R.height != h:
        R = R.resize((int(R.width * (h / R.height)), h))

    canvas = Image.new("RGB", (L.width + R.width, h), (255, 255, 255))
    canvas.paste(L, (0, 0))
    canvas.paste(R, (L.width, 0))
    canvas.save(out_png)


def write_env():
    lines = []
    lines += ["=== System ===", cmd_out(["uname", "-a"]), ""]
    lines += ["=== Python ===", cmd_out(["python", "-V"]), ""]
    lines += ["=== pip ===", cmd_out(["pip", "-V"]), ""]
    lines += ["=== nvidia-smi ===", cmd_out(["nvidia-smi"]), ""]
    with open(ENV_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def fmt(x):
    if x is None:
        return "—"
    return f"{x:.6g}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.isdir(VE_DIR):
        raise FileNotFoundError(f"Missing VE artifacts dir: {VE_DIR}")
    if not os.path.isdir(VP_DIR):
        raise FileNotFoundError(f"Missing VP artifacts dir: {VP_DIR}")

    # report_*.npz should have IS/fid/kid keys
    ve_report = pick_one(os.path.join(VE_DIR, "**", "report_*.npz"))
    vp_report = pick_one(os.path.join(VP_DIR, "**", "report_*.npz"))
    if not ve_report or not vp_report:
        raise FileNotFoundError("Missing report_*.npz in VE/VP artifacts dirs.")

    # prefer samples_0.npz if exists
    ve_samples = pick_one(os.path.join(VE_DIR, "**", "samples_*.npz"), prefer_basename="samples_0.npz")
    vp_samples = pick_one(os.path.join(VP_DIR, "**", "samples_*.npz"), prefer_basename="samples_0.npz")
    if not ve_samples or not vp_samples:
        raise FileNotFoundError("Missing samples_*.npz in VE/VP artifacts dirs.")

    ve_IS, ve_fid, ve_kid = load_report_metrics(ve_report)
    vp_IS, vp_fid, vp_kid = load_report_metrics(vp_report)

    # Summary CSV (2 rows)
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "IS", "fid", "kid", "report_path", "samples_path"])
        w.writerow([VE_RUN, ve_IS, ve_fid, ve_kid, ve_report, ve_samples])
        w.writerow([VP_RUN, vp_IS, vp_fid, vp_kid, vp_report, vp_samples])

    # Images
    save_grid(ve_samples, IMG_VE, n=64, cols=8)
    save_grid(vp_samples, IMG_VP, n=64, cols=8)
    side_by_side(IMG_VE, IMG_VP, IMG_SIDE)

    # Env
    write_env()

    # Markdown report
    md = []
    md += ["# Engineering Validation Report (CIFAR-10, pretrained checkpoints)", ""]
    md += ["## Runs", ""]
    md += [f"- VE: `{VE_RUN}`", f"- VP: `{VP_RUN}`", ""]
    md += ["## Metrics (IS / FID / KID)", ""]
    md += [f"- **VE** IS={fmt(ve_IS)}, FID={fmt(ve_fid)}, KID={fmt(ve_kid)}"]
    md += [f"- **VP** IS={fmt(vp_IS)}, FID={fmt(vp_fid)}, KID={fmt(vp_kid)}", ""]
    md += ["> IS higher is better; FID/KID lower is better.", ""]
    md += ["## Samples (grid)", ""]
    md += [f"![VE vs VP]({os.path.basename(IMG_SIDE)})", ""]
    md += ["## Outputs", ""]
    md += [f"- Summary CSV: `{os.path.basename(CSV_PATH)}`"]
    md += [f"- Env snapshot: `{os.path.basename(ENV_TXT)}`"]
    md += [f"- VE grid: `{os.path.basename(IMG_VE)}`"]
    md += [f"- VP grid: `{os.path.basename(IMG_VP)}`", ""]

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("[OK] Wrote:", CSV_PATH)
    print("[OK] Wrote:", REPORT_MD)
    print("[OK] Wrote:", ENV_TXT)
    print("[OK] Wrote:", IMG_VE)
    print("[OK] Wrote:", IMG_VP)
    print("[OK] Wrote:", IMG_SIDE)


if __name__ == "__main__":
    main()