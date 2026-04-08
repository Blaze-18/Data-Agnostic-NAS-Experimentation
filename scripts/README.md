Scripts and usage

- `move_dataset_to_data.ps1`: Moves original .pth from `Dataset/` into `data/raw/`.
- `split_pth_into_chunks.py`: Split a large .pth into smaller chunk files (place in `scripts/`).

Typical local workflow (PowerShell):

```powershell
# Activate environment
& "F:\Thesis\Experimentation\envs\nasbench_env\Scripts\Activate.ps1"

# Move dataset into standardized location
cd F:\Thesis\Experimentation
.
\scripts\move_dataset_to_data.ps1

# Split into ~200MB chunks
python scripts\split_pth_into_chunks.py --input data\raw\NAS-Bench-201-v1_0-e61699.pth --outdir data\chunks --max-bytes 200000000
```
