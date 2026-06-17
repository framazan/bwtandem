# BWTandem Benchmarking Results (Updated)

Here are the compiled benchmarking results across all three experiments, updated with new precision, coverage, boundary offset, and normalized fragmentation metrics, as well as distinct TRASH evaluation categories.

## Experiment 1: General Tandem Repeat Detection (Human GRCh38)
*Note: NCRF is motif-guided and skipped for de novo detection. TideHunter was also omitted per the plan.*

| Tool         |   Total Regions |   Adotto Recall (%) |   Adotto Precision (%) |   BP Recall (%) |   BP Precision (%) |   Unique Regions |   Runtime (s) |   Memory (GB) |
|:-------------|----------------:|--------------------:|-----------------------:|----------------:|-------------------:|-----------------:|--------------:|--------------:|
| trf          |         1011405 |               41.33 |                   0.11 |           44.66 |               0.09 |            21518 |     121426    |          1.45 |
| mreps        |        18676554 |                0    |                   0    |            0    |               0    |         17341714 |       3281.08 |          6.38 |
| ultra        |         3347214 |               80.85 |                   0.04 |           47.23 |               0.09 |           852377 |     107209    |          1.68 |
| bwtandem     |          695062 |               36.87 |                   0.1  |           34.3  |               0.08 |            67900 |      17566    |         41.98 |
| tantan       |         3469229 |               75.17 |                   0.05 |           31.06 |               0.11 |           725175 |       3106.94 |          0.27 |
| trash_denovo |            5329 |                7.12 |                   0.38 |           17.78 |               0.06 |                1 |     387456    |         14.59 |

---

## Experiment 2: Centromere Detection in Arabidopsis (Col-CEN)

| Tool           |   Regions |   Centromere Cov (%) |   CEN180 Count |   Runtime (s) |   Memory (GB) |
|:---------------|----------:|---------------------:|---------------:|--------------:|--------------:|
| trf            |     27626 |                84.39 |            191 |     475020    |          1.26 |
| mreps          |    342377 |                 1.58 |              0 |        848.71 |          0.84 |
| ultra          |    150788 |                 2.25 |              0 |       4802    |          1.68 |
| bwtandem       |     30329 |                81.74 |            219 |        705.2  |          5.32 |
| tantan         |    139086 |                 1.04 |              0 |        137.79 |          0.04 |
| ncrf           |        74 |                85.17 |              0 |       2863.63 |         80.96 |
| trash_template |       591 |                85.03 |            189 |      91360    |          2.63 |

---

## Experiment 3: Maize (T2T Mo17)

### 3A. Microsatellite Detection

| Tool           |   Total SSR bp |   Regions |   Runtime (s) |   Memory (GB) |
|:---------------|---------------:|----------:|--------------:|--------------:|
| trf            |        4827610 |     29559 |      18781    |          1.2  |
| ultra          |              0 |         0 |        827.86 |          2.23 |
| bwtandem       |        3044731 |     54283 |       3190.52 |         45.61 |
| tantan         |       55842884 |   1303967 |       1973.47 |          0.5  |
| ncrf           |        2604968 |       214 |        473.91 |         17.47 |
| trash_denovo   |       86876692 |      5368 |     215469    |         12.78 |
| trash_template |       47525434 |      3016 |     215469    |         12.78 |

### 3B. Satellite Detection (knob180/TR-1)

| Tool           |   knob180 arrays (of 25) |   TR-1 arrays (of 17) |   knob180 Norm Frag Score |   knob180 Mean Offset Error (bp) |   Runtime (s) |   Memory (GB) |
|:---------------|-------------------------:|----------------------:|--------------------------:|---------------------------------:|--------------:|--------------:|
| trf            |                       25 |                    17 |                      0    |                           651.96 |      19851    |          1.2  |
| ultra          |                        0 |                     0 |                      0    |                             0    |     125036    |          9.68 |
| bwtandem       |                       25 |                    16 |                      0    |                           985.06 |      13772    |         47.01 |
| tantan         |                       24 |                    17 |                      0    |                          7065.4  |       1968.4  |          0.5  |
| ncrf           |                        0 |                     0 |                      0    |                             0    |          2.25 |          0.29 |
| trash_denovo   |                       25 |                    17 |                      0.02 |                         10086.2  |     366282    |        120.64 |
| trash_template |                       25 |                    17 |                      0.05 |                          9850.62 |     366282    |        120.64 |

### 3C. CentC Detection

| Tool           |   CentC arrays (of 17) |   CentC Norm Frag Score |   CentC Mean Offset Error (bp) |   Runtime (s) |   Memory (GB) |
|:---------------|-----------------------:|------------------------:|-------------------------------:|--------------:|--------------:|
| trf            |                     17 |                    0.01 |                          69.62 |      19736    |          1.2  |
| ultra          |                      0 |                    0    |                           0    |      46284    |          3.38 |
| bwtandem       |                     17 |                    0.01 |                        4837.53 |      15358    |         45.53 |
| tantan         |                     17 |                    0    |                        9243.5  |       2110.63 |          0.5  |
| ncrf           |                      0 |                    0    |                           0    |          1.83 |          0.29 |
| trash_denovo   |                     17 |                    0.05 |                        5318.71 |     351900    |         42.83 |
| trash_template |                     17 |                    0.05 |                        5318.71 |     351900    |         42.83 |
