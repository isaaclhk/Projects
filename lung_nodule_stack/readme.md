# Title: Radiomics-Based Lung Nodule Classification Using a Stacking Ensemble

## Team Members
- Shao Ming Koh
- Isaac Lam Hong Kei
- Nguyen Thi Thanh Mai

## Abstract
Radiomics, an emerging field in medical imaging, leverages advanced mathematical analysis to extract quantitative
metrics from medical images, aiding in the early detection, diagnosis, and treatment of lung cancer. This study
focuses on improving the risk prediction of small lung nodules using machine learning models. We employed a
stacking ensemble approach, integrating Principal Component Analysis (PCA) for dimensionality reduction and selected
features from the Small Nodule Radiomics-Predictive Vector (SN-RPV). Base models employed in the stacking
ensemble were Support Vector Machine (SVM), Random Forest, k-Nearest Neighbors (KNN), and Naive Bayes classifiers.
Despite the theoretical advantages of stacking ensembles, our models demonstrated poorer performance on the
test set compared to the simpler SN-RPV model by Hunter et al. This outcome highlights the challenges of overfitting
and underscores the importance of model simplicity and interpretability in clinical applications. Future research
should explore alternative regularization techniques to improve the generalization of complex ensemble methods.

## Data Access
The radiomics data used in this study is publicly accessible in the Mendeley database under the accession code [10.17632/rxn95mp24d.1](https://data.mendeley.com/datasets/rxn95mp24d/1).

## Installation and Usage
1. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
2. **Ensure that data is in the correct relative path:**
    ```python
    train_dir = 'data/Features_Train.csv'
    test_dir = 'data/Features_Test.csv'
    ```
3. **Run the script**\
    To execute the baseline model:
    ```bash
    python replicate_hunters.py
    ```
    To run the stacking ensemble pipelines:
    ```bash
    python stack.py
    ```

## File Descriptions

### Scripts
- **replicate_hunters.py**: The python script to reproduce the baseline model.
- **stack.py**: Executes the stacking ensemble pipelines.
- **config.py**: Stores configuration settings and parameters used throughout the project.
- **evaluate_utils.py**: Includes functions for evaluating the performance of the model.

### Folders
- **images**: Contains the pipeline flowchart, and AUROC plots generate from the scripts.

## Results
For detailed results and analysis, please refer to the accompanying paper.

## Visualizations
Plots and visualizations are saved in the `images` folder.

## Acknowledgements
All authors contributed equally to this study.

## Citation
Please cite the following article: https://doi.org/10.1101/2025.04.28.25326620.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.