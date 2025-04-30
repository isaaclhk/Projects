# Title: Radiomics-Based Lung Nodule Classification Using a Stacking Ensemble

### Team Members
- Shao Ming Koh
- Isaac Lam Hong Kei
- Nguyen Thi Thanh Mai

### Description
This project focuses on classifying lung nodules using radiomics features and a stacking ensemble method. For detailed information, please refer to the accompanying paper.

### Installation and Setup
1. **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
2. **Ensure that data is in the correct relative path:**
    ```python
    train_dir = 'data/Features_Train.csv'
    test_dir = 'data/Features_Test.csv'
    ```

### Usage
- **Hunter's SN-RPV:** Execute the script `replicate_hunters.py`.
- **Stacking Ensemble:** Execute the script `stack.py`.
- **Hyperparameter Tuning:** Adjust the range of hyperparameters in `config.py`.
- **Evaluation Methods:** Methods for evaluation are stored in `evaluate_utils.py`.

### Results
For detailed results and analysis, please refer to the accompanying paper.

### Visualizations
Plots and visualizations are saved in the `images` folder.

### Acknowledgements
All authors contributed equally to this project.

### License
This project is licensed under the MIT License. See the LICENSE file for more details.