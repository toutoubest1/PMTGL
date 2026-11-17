#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

for file in isoprenoid_wille2004.csv eye_scheetz2006.csv; do
  if [[ ! -f "data/$file" ]]; then
    printf 'Missing data/%s. See data/README.md.\n' "$file" >&2
    exit 1
  fi
done

mkdir -p results paper/tables paper/figures
python -m pytest -q tests
python experiments/theory_figures.py
python experiments/psd_study.py --reps 15
python experiments/run_composite.py 10
python experiments/run_main.py a 20
python experiments/run_main.py b 10
python experiments/run_main.py c 5
python experiments/run_sweeps.py 20
python experiments/run_stars.py 10
python experiments/run_runtime.py
python experiments/real_sachs.py --reps 30
python experiments/sachs_full.py
python experiments/psd_table.py
python experiments/composite_table.py
python experiments/aux_tables.py
python experiments/run_main.py d 0
python experiments/real_stage6.py
python experiments/psd_stage6.py
PLUG="GLASSO,Kendall-GL,GaussRank-GL,Wrap-GL,Winsor-GL,MT-GLASSO(TH),PMT,PMT-mt"
python experiments/run_tuning2.py results/tuning_p100.csv 30 100 100 er2 clean,t3,row,homcell,hetcell,hetcell_asym,hetcell_mod "$PLUG"
python experiments/run_tuning2.py results/tuning_p200.csv 10 100 200 er2 clean,homcell,hetcell "$PLUG"
for g in hub band; do
  python experiments/run_tuning2.py results/tuning_graphs.csv 10 100 100 "$g" clean,homcell,hetcell "GLASSO,Kendall-GL,Wrap-GL,Winsor-GL,PMT,PMT-mt"
done
python experiments/run_tuning2.py results/tuning_p100_iter.csv 10 100 100 er2 clean,row,homcell,hetcell tlasso,DPD-GL
python experiments/main_tables.py
python experiments/paired_main.py
python experiments/real_tables_stage6.py
python experiments/psd6_table.py
python experiments/tuning_tables.py
