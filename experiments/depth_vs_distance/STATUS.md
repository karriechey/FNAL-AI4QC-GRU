# Depth vs distance: status

Updated 2026-11-01 04:28 UTC by dvd.py report.

- d5 peak LR: 3e-3 (highest LR stable at L = 1 and L = 5 (stable: ['1e-3', '2e-3', '3e-3']))
- d5 L_min: 1 (smallest L with ratio < 1 and McNemar p < 0.01); second seeds: ['dvd_d5_L01_u84_s1']
- d7 peak LR: 1e-3 (highest LR stable at L = 1 and L = 5 (stable: ['1e-3']))
- d7 L_min: 2 (smallest L with ratio < 1 and McNemar p < 0.01); second seeds: ['dvd_d7_L01_u168_s1', 'dvd_d7_L02_u168_s1']

## Eval block, GRU / MWPM

| d | L=1 | L=2 | L=3 | L=4 | L=5 |
|---|---|---|---|---|---|
| 5 | 0.774 (p 1.8e-208); seed 1 0.765 | 0.675 (p <1e-300) | 0.680 (p <1e-300) | 0.653 (p <1e-300) | 0.688 (p <1e-300) |
| 7 | 1.413 (p 6.1e-202) | 0.711 (p 1.1e-157) | 0.710 (p 4.1e-159) | 0.614 (p 1.4e-296) | 0.576 (p <1e-300) |
| 9 |  |  |  |  |  |

## Done (24)

- lrscan_d5_L05_lr1e-3 (DONE)
- lrscan_d5_L05_lr2e-3 (DONE)
- lrscan_d5_L05_lr3e-3 (DONE)
- lrscan_d5_L01_lr1e-3 (DONE)
- lrscan_d5_L01_lr2e-3 (DONE)
- lrscan_d5_L01_lr3e-3 (DONE)
- dvd_d5_L05_u84 (DONE)
- dvd_d5_L04_u84 (DONE)
- dvd_d5_L03_u84 (DONE)
- dvd_d5_L02_u84 (DONE)
- dvd_d5_L01_u84 (DONE)
- lrscan_d7_L05_lr1e-3 (DONE)
- lrscan_d7_L05_lr2e-3 (DONE)
- lrscan_d7_L05_lr3e-3 (KILLED)
- lrscan_d7_L01_lr1e-3 (DONE)
- lrscan_d7_L01_lr2e-3 (DONE)
- lrscan_d7_L01_lr3e-3 (DONE)
- dvd_d7_L05_u168 (DONE)
- dvd_d7_L04_u168 (DONE)
- dvd_d7_L03_u168 (DONE)
- dvd_d7_L02_u168 (DONE)
- dvd_d7_L01_u168 (DONE)
- lrscan_d9_L05_lr1e-3 (DONE)
- dvd_d5_L01_u84_s1 (DONE)

## Running (2)

- dvd_d7_L01_u168_s1: epoch 35, about 2.5 h left
- dvd_d7_L02_u168_s1: epoch 18, about 5.0 h left

## Queued (0)


## Paused (10), distances in queue/paused_d; resume from the last snapshot

- lrscan_d9_L05_lr2e-3
- lrscan_d9_L05_lr3e-3
- lrscan_d9_L01_lr1e-3
- lrscan_d9_L01_lr2e-3
- lrscan_d9_L01_lr3e-3
- dvd_d9_L05_u280
- dvd_d9_L04_u280
- dvd_d9_L03_u280
- dvd_d9_L02_u280
- dvd_d9_L01_u280

Remaining run-hours about 8; at the current 2 concurrent runs the queue needs roughly 4 h more (ETA about 2026-11-01 08:13 UTC). Second-seed runs are not counted until L_min is known; paused runs are not counted.
