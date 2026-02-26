# Redox Potential Prediction for Iron-Sulfur Proteins

This repository contains the data and code for predicting redox potentials of iron-sulfur (Fe-S) proteins using machine learning (ML), as detailed in my bachelor thesis, *Machine Learning for Prediction of Redox Potentials in Iron-Sulfur Proteins*, provided here as `BEP_report.pdf`. 
The project is directly inspired by Galuzzi et al.'s work on [Flavoprotein redox potential prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00858), and leverages structural and physicochemical features extracted from AlphaFold-predicted structures to train ML models, achieving state-of-the-art performance for Fe-S proteins.

## Overview

Iron-sulfur proteins are critical metalloproteins involved in redox reactions across all domains of life. Their midpoint redox potentials ($E_m$) are challenging to predict due to structural complexity and diverse physicochemical environments. This project develops an ML framework to predict $E_m$ using a dataset of 168 entries from 130 proteins across four cofactor types ([4Fe-4S], [3Fe-4S], [2Fe-2S], Fe3+), compiled from 113 publications. Features include structural (e.g., burial depth, HiPIP indicators) and physicochemical properties (e.g., hydrophobicity, volume) extracted at various radii around the cofactor.

### Key findings:
- **Models**: Gradient Boosting (XGB) and Random Forest (RF) outperformed linear and kernel-based models, with XGB achieving mean absolute errors (MAEs) of 84.09 mV (radius-independent) and 88.17 mV (radius-dependent), and RF achieving 61.49 mV on the [2Fe-2S] dataset with bar features.
- **SHAP Analysis**: Identified hydrophobicity, burial depth, and HiPIP-specific features as critical predictors, reflecting their role in modulating electron transfer.
- **Comparison**: MAEs were 69% higher than Galuzzi et al.’s flavoprotein benchmark (36.4 mV) [1], but relative errors (6.83–7.65%) were comparable despite a broader redox range (1196 mV vs. 472.5 mV).
- **Limitations**: Dataset heterogeneity, AlphaFold inaccuracies (e.g., P73276, RMSD 14Å), pH biases (30/38 high pH in [4Fe-4S]), and computational constraints.

## User manual

The detailed instructions on using this repository can be found in the document `detailed_methodology.pdf` (Appendix E of the `BEP_report.pdf` document).

## Fe-S Protein Dataset

The dataset is (168 entries) is available at `FeS_data/FeS_raw_data/complete_FeS_dataset_with_cofactor_id.xlsx`. For each Fe-S protein used in this work, I most importantly report PDB-ID and reference of the experimental work.

## Repository Organization

- `FeS_data/final_structure_dataset/`: the structures with cofactors of all protein entries
- `energy_minimization/`: scripts to induce mutations into structures and energy-minimize them
- `cofactor_implanting/`: scripts to download structures with cofactors from AlphaFill and to implant cofactors into empty structures, and folders with inputs and outputs of these scripts
- `feature_extraction/`: scripts to extract features from protein structures, merge with experimental pH data, and create data subsets of features. Outputs in `FeS_data/data_extracted_features`.
- `em_prediction_ml_training`: ML training scripts and supercomputer script for running them. Outputs in `ml_training_outputs`.
- `results_analysis-visualization`: various result analysis and visualization scripts and their outputs in separate folders per category. 
- `virtual_environments`: yaml files with all virtual environments needed to run this pipeline. 

Details on all other files can be found in the Detailed Methodology.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/laetichou/FeS_Em_Prediction.git 
   ```

2. **Set Up Environments**:
I recommend using mamba as a package manager.
- Install [mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
- Navigate to cloned repository
   ```bash
   cd FeS_Em_Prediction
   ```
- Create environments with the provided yaml files. For each file, run:
   ```bash
   mamba env create -f virtual_environments/environment_name.yml
   ```

3. **Download Dataset**:
   - The protein structures (wild-type, and computationally-minimized mutant structures) are provided in `FeS_data/final_structure_dataset`
   - The extracted features are provided in `FeS_data/final_structure_dataset`, so the pre-processing step can be skipped.
   - If structures need to be added to the dataset before extracting features and running models, this can be done by following sections **E.1** to **E.4** in the Detailed Methodology. 


## Contact

For comments and questions, contact [l.a.e.guerin@student.tudelft.nl](mailto:l.a.e.guerin@student.tudelft.nl).
