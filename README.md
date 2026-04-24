# saramorg_fret_nroot
Analysis of Arabidopsis Thaliana roots, FRET-ratio in nuclei compartment. 3D reconstruction of root structure.

## Batch Processing CLI

Once inside the repo root activate the pixi environment using:

`pixi shell`

You can run the batch LIF processing pipeline from command line using:

`python src/run_batch_processing.py --input-dir <path_to_raw_data> --config <path_to_config_yaml>`

### Required arguments

- `--input-dir`: directory containing `.lif` containers to process.
- `--config`: YAML file with model, inference, segmentation, and batch settings.

### Example

`python src/run_batch_processing.py --input-dir C:\Users\adiez_cmic\github_repos\saramorg_fret_nroot\raw_data --config configs/batch_processing.example.yaml`

### Config template

An annotated example config is available at:

- `configs/batch_processing.example.yaml`

This file documents each supported variable and mirrors the same parameters used in the batch notebook.
