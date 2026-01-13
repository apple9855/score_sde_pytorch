# Engineering Validation Report (CIFAR-10, pretrained checkpoints)

## Runs

- VE: `VE_ncsnpp_ckpt24`
- VP: `VP_ddpmpp_ckpt26`

## Metrics (IS / FID / KID)

- **VE** IS=8.56561, FID=175.819, KID=nan
- **VP** IS=6.9466, FID=163.553, KID=nan

> IS higher is better; FID/KID lower is better.

## Samples (grid)

![VE vs VP](ve_vs_vp_side_by_side.png)

## Outputs

- Summary CSV: `summary.csv`
- Env snapshot: `RUN_ENV.txt`
- VE grid: `ve_samples_grid.png`
- VP grid: `vp_samples_grid.png`
