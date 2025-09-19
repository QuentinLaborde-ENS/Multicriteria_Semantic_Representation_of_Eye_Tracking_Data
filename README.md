# Multicriteria Semantic Representation of Eye-Tracking Data

Code accompanying the paper: **Multicriteria Semantic Representation of Eye-Tracking Data Using Adaptive Symbolization**.  
This repository implements the full, interpretable pipeline to convert multivariate eye-tracking features into symbolic sequences, compute semantic distances between recordings, and evaluate downstream tasks.

<p align="center">
  <img src="docs/figures/pipeline_overview.png" alt="Pipeline overview" width="720">
</p>

## 🌐 Overview

Per semantic dimension (fixations, saccades, scanpaths, AoIs), the pipeline performs:
1. **ECDF normalization** → values mapped to \[0,1]  
2. **Adaptive segmentation (PELT)** → piecewise-constant mean segments  
3. **Symbolization (Kernel PCA → K-Means)** → each centroid defines a symbol  
4. **Sequence distance (Wagner–Fischer)** → substitution cost = distance between symbol centroids  
5. **Fusion** of per-dimension distance matrices    
6. **Evaluation** via **MDS** + **SVM**

The goal is a **semantic, interpretable, multi-criteria representation** of eye-tracking behavior.

---

## 📦 Repository structure

Multicriteria_Semantic_Representation_of_Eye_Tracking_Data/
├── main.py # Main entry point
├── configurations/ # YAML configs per dataset
│ ├── analysis_etra.yaml
│ ├── analysis_gazebase.yaml
│ └── analysis_cldrive.yaml
├── processing/ # Core modules
│ ├── normalization.py
│ ├── segmentation.py
│ ├── symbolization.py
│ ├── clustering.py
├── input/ # Visual feature series obtained from raw eye-tracking data
│ ├── ETRA/
│ ├── GazeBase/
│ └── CLDrive/
├── output/ # Results saved here
│ ├── ETRA/
│ ├── GazeBase/
│ └── CLDrive/
├── environment.yml # Conda environment file
└── README.md


## ⚙️ Installation

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate gaze-symbols
```bash


## 🚀 Usage

After installing the dependencies (see [Installation](#-installation)), you can run the main pipeline directly from the command line.

### Example

```bash
python main.py --ETRA
```bash
This will execute the pipeline on the ETRA dataset using the default symbolization method (Kernel PCA).

### Command-line Arguments

You can specify the dataset and the method as command-line arguments:
- `--ETRA` : use the **ETRA** dataset  
- `--CLDRIVE` : use the **CLDrive** dataset  
- `--GAZEBASE` : use the **GazeBase** dataset 


## 📖 Citation

If you use this code or find our work useful in your research, please cite:

```bibtex
@article{laborde2025multicriteria,
  title={A Multicriteria Semantic Representation of Eye-Tracking Data},
  author={Laborde, Quentin and [co-authors if any]},
  journal={Pre-print}
}
```bibtex



