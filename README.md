# Multicriteria Semantic Representation of Eye-Tracking Data

Code accompanying the paper: **Multicriteria Semantic Representation of Eye-Tracking Data Using Adaptive Symbolization**.  
This repository implements the full, interpretable pipeline to convert multivariate eye-tracking features into symbolic sequences, compute semantic distances between recordings, and evaluate downstream tasks.

 

## 🌐 Overview

Per semantic dimension (fixations, saccades, scanpaths, AoIs), the pipeline performs:
1. **Data normalization** → values mapped to \[0,1]  
2. **Adaptive segmentation (PELT)** → piecewise-constant multivariate segments  
3. **Symbolization (Kernel PCA → K-Means)** → each centroid defines a symbol  
4. **Sequence distance (Wagner–Fischer)** → substitution cost = distance between symbol centroids  
5. **Fusion** of per-dimension distance matrices    
6. **Clustering** via **MDS** + **SVM**

The goal is a **semantic, interpretable, multi-criteria representation** of eye-tracking behavior.

---

 
## ⚙️ Installation

### Using conda (recommended)

```bash
conda env create -f environment.yml
conda activate gaze-symbols
```


## 🚀 Usage

After installing the dependencies (see [Installation](#-installation)), you can run the main pipeline directly from the command line.

### Example

```bash
python main.py --ETRA
```
This will execute the pipeline on the ETRA dataset using the default symbolization method (Kernel PCA).

### Command-line Arguments

You can specify the dataset as command-line arguments:
- `--ETRA` : use the **ETRA** dataset  
- `--CLDRIVE` : use the **CLDrive** dataset  
- `--GAZEBASE` : use the **GazeBase** dataset 


## 📖 Citation

If you use this code or find our work useful in your research, please cite:

```bibtex
@article{laborde2025multicriteria,
  title={A Multicriteria Semantic Representation of Eye-Tracking Data Using Adaptive Symbolization},
  author={Laborde, Quentin and Laurent Oudre and Nicolas Vayatis and Ioannis Bargiota},
  journal={Pre-print}
}
```



