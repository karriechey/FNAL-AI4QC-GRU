# u70 GRU ReuseFactor sweep: upper-bound baseline, <32,16> PTQ

hls4ml `2ed80fb9a6d5150a7f1ec21a8cf6564900244a0e` (main). Config: Resource, io_parallel, static GRU, xcvu13p-flga2577-2-e, 5.0 ns, ap_fixed<32,16>. Resources are Vivado HLS 2020.1 C-synthesis estimates; % uses the device totals in the same report. Not deployment numbers.

| requested RF | local patch | effective RF (in / rec / dense) | latency cycles | latency µs @5 ns | II | DSP | LUT | FF | BRAM18K | est. clock ns | timing violation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | none | 24 / 35 / 35 | 560-573 | 2.8-2.865 | 524-536 | 3362 (27.36%) | 540780 (31.3%) | 119442 (3.46%) | 667 (12.41%) | 4.328 | False |
| 70 | kernelfix | 72 / 70 / 70 | 1027-1040 | 5.135-5.2 | 956-968 | 1958 (15.93%) | 434343 (25.14%) | 96256 (2.79%) | 355 (6.6%) | 4.328 | False |
| 140 | none | 144 / 140 / 70 | 1879-1892 | 9.395-9.46 | 1808-1820 | 1398 (11.38%) | 424910 (24.59%) | 91849 (2.66%) | 230 (4.28%) | 4.328 | False |
