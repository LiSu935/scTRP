# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is not an application codebase — it's a small set of Bash/SLURM scripts (`hellbender_trust4/`) that drive [TRUST4](https://github.com/liulab-dfci/TRUST4) (a TCR/BCR repertoire reconstruction tool) over 15 single-cell RNA-seq samples on the University of Missouri's Hellbender HPC cluster. There is no application source, package manifest, or test suite to build/lint/run — "development" here means editing the shell scripts and the manifest, then re-running (or dry-checking) them.

The scripts assume a specific remote layout that does not exist locally:

```text
/cluster/pixstor/xudong-lab/wangdu/scTRP/
├── cd8_fastq_all/              # per-sample FASTQ inputs (not in this repo)
├── hellbender_trust4/          # this repo
└── tools/TRUST4_v1.1.9/        # TRUST4 source, compiled in place (not in this repo)
```

All scripts here take/derive `PROJECT_DIR` = that top-level directory (default `$PWD`), not the repo root.

## The pipeline (in order)

1. **Sync files to Hellbender** — `rsync` FASTQs, TRUST4 source, and this repo to the cluster (see [README_hellbender_steps.md](README_hellbender_steps.md) for the exact commands/paths; they embed a specific user's local Windows path and NetID and will need adjusting per-user).
2. **Compile TRUST4** — [build_trust4.sh](build_trust4.sh) `PROJECT_DIR`: runs `make -j 4` inside `tools/TRUST4_v1.1.9`, then asserts `run-trust4`, `trust4`, `human_IMGT+C.fa`, `human_vdjc.list` exist. Requires a C++ toolchain (`module load gcc` if `make`/`g++` aren't on PATH).
3. **Submit the SLURM job array** — either directly:
   ```bash
   mkdir -p logs
   PROJECT_DIR=/cluster/pixstor/xudong-lab/wangdu/scTRP sbatch hellbender_trust4/submit_trust4_cd8_array.sbatch
   ```
   or via the wrapper [submit_from_project_dir.sh](submit_from_project_dir.sh), which hardcodes `PROJECT_DIR` and `cd`s there before calling `sbatch`.
4. **Check status**: `squeue -u "$USER"`, inspect `trust4_cd8_outputs/*/`, tail `logs/trust4_<jobid>_<taskid>.err`.

## Manifest-driven array jobs

[submit_trust4_cd8_array.sbatch](submit_trust4_cd8_array.sbatch) is a `--array=1-15` job. Each array task reads one line from [trust4_cd8_manifest.tsv](trust4_cd8_manifest.tsv) via:

```bash
awk -v task="${SLURM_ARRAY_TASK_ID}" 'NR == task + 1 {print; exit}' "${MANIFEST}"
```

i.e. task ID `N` maps to manifest line `N+1` (line 1 is the TSV header: `sample  srr  r1_barcode_umi  r2_cdna`). **If you add/remove/reorder manifest rows, the `--array` range in the sbatch script must be updated to match the new row count**, or extra tasks will fail the `No manifest row for SLURM_ARRAY_TASK_ID` check and missing tasks will silently skip samples.

Per task, the script:
- Resolves `R1` (barcode+UMI read) and `R2` (cDNA read) from the manifest's relative paths, joined to `PROJECT_DIR`.
- Runs `run-trust4` with `--readFormat "r1:0:-1,bc:0:15,um:16:-1"` — i.e. R1 is a 16bp cell barcode (positions 0-15) + 10bp UMI (positions 16 to end); R2 is the full cDNA read used for reconstruction. This encodes 10x Genomics-style barcode/UMI layout and must stay in sync with whatever upstream demux step produced these FASTQs.
- Writes output to `trust4_cd8_outputs/<sample>/` with file prefix `<srr>_<sample>_CD8_TRUST4`.
- Only reconstructs from scRNA-seq CD8 FASTQs — matched bulk TCR-seq is intentionally not used in this pipeline.

Key outputs per sample (in `trust4_cd8_outputs/<sample>/`): `*_cdr3.out`, `*_report.tsv`, `*_airr.tsv`, `*_barcode_report.tsv`, `*_barcode_airr.tsv`. The barcode-level files are what downstream analysis uses to join reconstructed TCRs back to h5ad cell barcodes.

## Running on a new dataset

1. **Compile TRUST4 (skip if already built)** — only needed once per cluster/environment:
   ```bash
   bash hellbender_trust4/build_trust4.sh "$PWD"
   ```

2. **Upload the new FASTQs to Hellbender.** Each sample needs an R1 (cell barcode + UMI) and R2 (cDNA) file:
   ```bash
   rsync -avP /local/path/to/new_fastqs/ YOUR_NETID@hellbender-login:/cluster/pixstor/xudong-lab/wangdu/scTRP/<new_fastq_dir>/
   ```

3. **Write a new manifest TSV**, using [trust4_cd8_manifest.tsv](trust4_cd8_manifest.tsv) as a template — header `sample	srr	r1_barcode_umi	r2_cdna`, one row per sample, with the FASTQ paths given relative to `PROJECT_DIR`.

4. **Update (or duplicate) the sbatch array script.** In [submit_trust4_cd8_array.sbatch](submit_trust4_cd8_array.sbatch):
   - `#SBATCH --array=1-15` must equal the number of data rows in the new manifest (not counting the header) — see "Manifest-driven array jobs" above for why a mismatch causes failed or silently skipped tasks.
   - `MANIFEST=` should point at the new TSV (and `OUT_ROOT=` at a new output directory, if you don't want to mix outputs with the CD8 run).
   - Prefer copying the script (e.g. `submit_trust4_new_array.sbatch`) over editing in place if you want to keep the original CD8 pipeline reproducible/re-runnable.

5. **Confirm the barcode/UMI read format still applies.** `--readFormat "r1:0:-1,bc:0:15,um:16:-1"` assumes a 16bp barcode + 10bp UMI (10x-style) on R1. Update this string if the new dataset uses different chemistry, or TRUST4 will misparse cell barcodes.

6. **Submit:**
   ```bash
   mkdir -p logs
   PROJECT_DIR=/cluster/pixstor/xudong-lab/wangdu/scTRP sbatch hellbender_trust4/submit_trust4_new_array.sbatch
   ```

7. **Monitor and collect outputs:**
   ```bash
   squeue -u "$USER"
   tail -n 50 logs/trust4_*_*.err
   ls -lh <OUT_ROOT>/*/
   ```
   Use `*_barcode_report.tsv` / `*_barcode_airr.tsv` per sample for downstream joins against h5ad cell barcodes.

## Working on these scripts

- All scripts use `set -euo pipefail` — preserve that when editing.
- `PROJECT_DIR` is always resolved as `${1:-$PWD}` or `${PROJECT_DIR:-$PWD}`, never hardcoded inside the array sbatch script itself (only the `submit_from_project_dir.sh` convenience wrapper hardcodes a path, for one specific user's environment).
- There's no local way to execute `run-trust4` or `sbatch` — these only run on Hellbender. Treat script edits as changes to verify by careful reading (paths, quoting, array bounds, manifest column order) rather than by local execution.
