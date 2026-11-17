# Real-data inputs

| Study | Files in this directory | Source |
| --- | --- | --- |
| Sachs et al. (2005) | `cyto_full_data.csv`, `cyto_full_target.csv` (included) | [Causal Discovery Toolbox resources](https://github.com/FenTechSolutions/CausalDiscoveryToolbox/tree/master/cdt/data/resources). See [CDT-LICENSE](CDT-LICENSE) for the upstream MIT notice. |
| Wille et al. (2004) | `isoprenoid_wille2004.csv` (local input) | [Original study and supplementary data](https://doi.org/10.1186/gb-2004-5-11-r92). The analysis uses the standardized 118-by-39 expression matrix from the research archive. |
| Scheetz et al. (2006) | `eye_scheetz2006.csv` (local input) | [CRAN flare](https://cran.r-project.org/package=flare), dataset `eyedata` (200 probe columns from `x` and `TRIM32` from `y`, 120 rows). |

For exact reproduction, copy both local inputs from the research archive's
`pmtgl/data/` directory to this directory. Keep the header row and omit any
row index. Use the archived CSVs to preserve the order and labeling of columns.

The checksums of those two CSVs are listed below for reference.

```text
b37832e03a6bcfab7b6711edea7b1e0310e2e6ba5f97774826b8732028124ee8  isoprenoid_wille2004.csv
1b10a1e0526f3c84f98419055ba10817c430dd43a8673605231e0f2a002e9ffa  eye_scheetz2006.csv
```
