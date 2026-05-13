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

### Per-image CSV outputs

Written to `<results_root>/<lif_container_id>/<lif_image_name>.csv` (see the `results_root` comment block in the example YAML). Besides morphology, intensities, FRET, depth, and tissue columns, each file includes:

- **`tip_cell`**: binary flag for the nucleus chosen as the root tip (`calculate_distance_to_tip`).
- **`distance_to_tip`**: value in `[0, 1]` after normalizing centroid distances to that tip nucleus.
- **`input_img_shape`**: JSON list `[Z, Y, X]` matching the `nuclei_labels` array shape (useful for 3D visualization and analysis notebooks).

Tip distance and image shape are not configurable via YAML; they are always added when a CSV is (re)computed.
