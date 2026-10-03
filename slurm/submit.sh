#!/bin/bash
# Submit the whole campaign as a chain of dependent jobs. Run from the repo root
# on the login node after `python -m src.fetch --weights`.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate regime2
python -m src.jobs
rows() { echo $(( $(wc -l < "work/jobs/$1.csv") - 1 )); }

prep=$(sbatch --parsable slurm/00_prepare.sh)
tc=$(sbatch --parsable --dependency=afterok:$prep --array=0-$(( $(rows tune_cpu) - 1 ))%8 slurm/10_tune_cpu.sh)
tg=$(sbatch --parsable --dependency=afterok:$prep --array=0-3 slurm/11_tune_gpu.sh)
rc=$(sbatch --parsable --dependency=afterok:$tc --array=0-63%24 slurm/20_run_cpu.sh)
rg=$(sbatch --parsable --dependency=afterok:$tg --array=0-3 slurm/21_run_gpu.sh)
rf=$(sbatch --parsable --dependency=afterok:$prep --array=0-2 slurm/22_run_fm.sh)
an=$(sbatch --parsable --dependency=afterok:$rc:$rg:$rf slurm/30_analysis.sh)
echo "prepare $prep | tune $tc $tg | run $rc $rg $rf | analysis $an"
