# Run TRUST4 CD8 reconstruction on Hellbender

This runs TRUST4 using only scRNA-seq-derived CD8 FASTQs:

- `R2_cdna.fastq.gz`: sequence reads used for TCR/CDR3 reconstruction.
- `R1_barcode_umi.fastq.gz`: 16 bp cell barcode + 10 bp UMI used to assign reconstructed TCRs back to cells.

Matched TCR-seq is not used here.

## 1. Upload files from local Windows to Hellbender

Run from a local terminal that has `rsync` or `scp`.

```bash
rsync -avP "C:/Users/wangd/Documents/Tumor-reactive T cell prediction/cd8_fastq_all/" YOUR_NETID@hellbender-login:/cluster/pixstor/xudong-lab/wangdu/scTRP/cd8_fastq_all/
rsync -avP "C:/Users/wangd/Documents/Tumor-reactive T cell prediction/tools/TRUST4_v1.1.9/" YOUR_NETID@hellbender-login:/cluster/pixstor/xudong-lab/wangdu/scTRP/tools/TRUST4_v1.1.9/
rsync -avP "C:/Users/wangd/Documents/Tumor-reactive T cell prediction/hellbender_trust4/" YOUR_NETID@hellbender-login:/cluster/pixstor/xudong-lab/wangdu/scTRP/hellbender_trust4/
```

`rsync -P` supports resume/progress. If the connection breaks, rerun the same command.

## 2. Log in to Hellbender

```bash
ssh YOUR_NETID@hellbender-login
cd /cluster/pixstor/xudong-lab/wangdu/scTRP
```

Expected structure:

```text
cd8_fastq_all/
hellbender_trust4/
tools/TRUST4_v1.1.9/
```

## 3. Compile TRUST4 once

```bash
bash hellbender_trust4/build_trust4.sh "$PWD"
```

If `make` or `g++` is unavailable, load a compiler module first, for example:

```bash
module avail gcc
module load gcc
bash hellbender_trust4/build_trust4.sh "$PWD"
```

## 4. Submit all 15 samples

```bash
mkdir -p logs
PROJECT_DIR=/cluster/pixstor/xudong-lab/wangdu/scTRP sbatch hellbender_trust4/submit_trust4_cd8_array.sbatch
```

The job array is `1-15`, one task per sample.

## 5. Check status

```bash
squeue -u "$USER"
ls -lh trust4_cd8_outputs/*/
tail -n 50 logs/trust4_*_*.err
```

## 6. Important output files

Each sample writes to:

```text
trust4_cd8_outputs/<sample>/
```

Look especially for:

- `*_cdr3.out`
- `*_report.tsv`
- `*_airr.tsv`
- `*_barcode_report.tsv`
- `*_barcode_airr.tsv`

The barcode-level files are the key files for matching TRUST4-reconstructed TCRs back to h5ad cell barcodes.
