# CITL

## 📦 Dependencies and Versions

The following Python packages and versions were used in this project:

| Package               | Version          |
| --------------------- | ---------------- |
| python                | 3.10.16
| contourpy             | 1.3.1            |
| cuda-cudart           | 12.1.105         |
| cuda-cupti            | 12.1.105         |
| cuda-libraries        | 12.1.0           |
| cuda-nvrtc            | 12.1.105         |
| cuda-nvtx             | 12.1.105         |
| cuda-opencl           | 12.8.55          |
| cuda-runtime          | 12.1.0           |
| cuda-version          | 12.8             |
| cycler                | 0.12.1           |
| docker-pycreds        | 0.4.0            |
| einops                | 0.8.1            |
| ffmpeg                | 4.3              |
| filelock              | 3.11.0           |
| fonttools             | 4.56.0           |
| freetype              | 2.12.1           |
| gitdb                 | 4.0.12           |
| gitpython             | 3.1.44           |
| googledrivedownloader | 1.1.0            |
| grpcio                | 1.71.0           |
| h5py                  | 3.13.0           |
| idna                  | 3.7              |
| jinja2                | 3.1.5            |
| joblib                | 1.4.2            |
| json-tricks           | 3.17.3           |
| kiwisolver            | 1.4.8            |
| lxml                  | 5.3.1            |
| markdown              | 3.7              |
| markupsafe            | 3.0.2            |
| matplotlib            | 3.10.1           |
| mkl                   | 2023.1.0         |
| mkl-service           | 2.4.0            |
| networkx              | 3.4.2            |
| nibabel               | 5.3.2            |
| nilearn               | 0.11.1           |
| nni                   | 2.10.1           |
| numpy                 | 1.23.5           |
| nvidia-ml-py          | 12.570.86        |
| openh264              | 2.1.1            |
| openjpeg              | 2.5.2            |
| openssl               | 3.0.15           |
| packaging             | 24.2             |
| pandas                | 2.2.3            |
| pillow                | 11.1.0           |
| pip                   | 25.0             |
| prettytable           | 3.15.1           |
| protobuf              | 5.29.4           |
| psutil                | 7.0.0            |
| pydantic              | 2.10.6           |
| pytorch               | 2.1.0            |
| pytorch-cuda          | 12.1             |
| pytorch-mutex         | 1.0              |
| scikit-learn          | 1.6.1            |
| scipy                 | 1.10.1           |
| seaborn               | 0.13.2           |
| sentry-sdk            | 2.23.1           |
| tensorboard           | 2.19.0           |
| tensorboardx          | 2.6.2.2          |
| torch-cluster         | 1.6.2+pt21cu121  |
| torch-geometric       | 2.0.3            |
| torch-scatter         | 2.1.2+pt21cu121  |
| torch-sparse          | 0.6.18+pt21cu121 |
| torch-spline-conv     | 1.2.2+pt21cu121  |
| torchaudio            | 2.1.0            |
| torchtriton           | 2.1.0            |
| torchvision           | 0.16.0           |
| tqdm                  | 4.67.1           |
| wandb                 | 0.19.8           |
| yacs                  | 0.1.8            |


The project structure is organized as follows:


## 📂 Directory Description

- **imports/** — Contains dataset construction files.  
- **net/** — Includes neural network model implementations.  
- Files starting with **“01”** — Pre-trained model implementation.  
- Files starting with **“02”** — Pseudo-label prediction and Training .  
- Files starting with **“03”** — model training and classification.


## 🧩 Dynamic Functional Connectivity (dFC) Computation

The following code demonstrates how dynamic functional connectivity (dFC) matrices are computed using a sliding window approach with **nilearn**:

```python
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from nilearn import datasets, input_data, plotting
import os

# Define atlas file
atlas_dir = 'dos160_roi_atlas.nii'
masker = input_data.NiftiLabelsMasker(
    labels_img=atlas_dir,
    standardize=True,
    memory='nilearn_cache',
    n_jobs=-1
)

# Folder containing functional images
folder_path = ''

# Define sliding window parameters
window_size = 50
step_size = 1

start_index = 获取文件夹名字.file_names_without_extension.index('')

for file_name in 获取文件夹名字.file_names_without_extension[start_index:]:
    print(file_name)
    func_filename = ''  # Specify functional image file path
    if os.path.exists(func_filename):
        # Extract time series for each ROI
        time_series = masker.fit_transform(func_filename)
        time_series1 = time_series[:, :-1]

        # Initialize connectivity measure
        connectivity_measure = ConnectivityMeasure(kind="correlation")

        # Initialize sliding window indices
        start_idx = 0
        end_idx = window_size

        # Create output folder
        folder_path1 = os.path.join('', file_name)
        os.makedirs(folder_path1, exist_ok=True)

        # Compute dynamic functional connectivity
        while end_idx <= len(time_series1):
            window_time_series = time_series1[start_idx:end_idx, :]
            connectivity_matrix = connectivity_measure.fit_transform([window_time_series])[0]

            # Save connectivity matrix
            np.save(folder_path1 + '\\' + str(start_idx) + '.npy', connectivity_matrix)
            print(start_idx)

            # Update sliding window
            start_idx += step_size
            end_idx += step_size
    else:
        print("File not found, skipping:", file_name)
        continue
