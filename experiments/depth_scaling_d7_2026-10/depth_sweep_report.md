# Depth scaling d=7: decision rules

Sweep LR: 1e-3

- L=1 -> 2: ratio 3.4517 -> 1.2659 (gain +63.3%), McNemar p <1e-300
- L=2 -> 3: ratio 1.2659 -> 1.2267 (gain +3.1%), McNemar p 0.0014
- L=3 -> 4: ratio 1.2267 -> 0.8474 (gain +30.9%), McNemar p 1.3e-248
- L=4 -> 5: ratio 0.8474 -> 0.8068 (gain +4.8%), McNemar p 6.3e-06
- L=5 -> 6: ratio 0.8068 -> 0.7995 (gain +0.9%), McNemar p 0.4
- L=6 -> 7: ratio 0.7995 -> 0.7774 (gain +2.8%), McNemar p 0.0089
- L=7 -> 8: ratio 0.7774 -> 0.7765 (gain +0.1%), McNemar p 0.92
- L=8 -> 9: ratio 0.7765 -> 0.7311 (gain +5.9%), McNemar p 2.4e-08
- L=9 -> 10: ratio 0.7311 -> 0.7119 (gain +2.6%), McNemar p 0.015

L* = 5


- match_d7_L04_u172 (1.2919) vs L=4 stack (0.8474): McNemar p 9.2e-321: stack wins
- match_d7_L08_u255 (0.8503) vs L=8 stack (0.7765): McNemar p 5e-15: stack wins
