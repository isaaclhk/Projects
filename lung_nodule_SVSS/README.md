# Radiomics-Based Lung Nodule Classification with Stochastic Search Variable Selection and Bayesian Logistic Regression



## Abstract
Radiomics is an emerging field in medical science that involves extracting quantitative features from medical images using data characterization algorithms. 
These features, known as radiomic features, provide a quantitative approach to medical image analysis. 
In the context of lung cancer, radiomics-based approaches are transforming disease management by improving early detection, 
diagnosis, prognosis, and treatment decision-making. 
This study aimed to explore the utility of Bayesian methods, specifically Stochastic Variable Selection and Shrinkage (SVSS) and Bayesian logistic regression, 
in the radiomics-based classification of small lung nodules with limited training data. 
The Bayesian approach matched the performance of frequentist Lasso logistic regression on the test set, 
demonstrating its viability as an alternative approach. 
Annulus\_GLCM\_Entrop\_LLL was consistently identified as a feature positively influencing small lung nodule malignancy prediction across multiple models. 
This finding enhances confidence in the effect of this feature, suggesting that future Bayesian analyses can incorporate this information for greater reliability in feature selection and coefficient estimates. 
This study highlights the potential of Bayesian methods to address the challenges of limited data in medical image analysis, 
offering a robust alternative to traditional statistical approaches and contributing to improved clinical decision-making.

For more details, refer to the corresponding paper.

## Installation and Usage

1. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2. **Ensure that data is in the correct relative path:**\
You may define the desired paths in `config.py`. The default paths are:
    ```python
    # paths to training and test set
    paths = {
        'A_path': 'data/Features_Test.csv',
        'B_path': 'data/Features_Train.csv'
    }
    ```
3. **Run the project**:
    ```bash
    python main.py
    ```

    *Note: to view the plots, you need to run the file in an interactive window.*

## File Descriptions

### Scripts

- **main.py**: This is the entry point of the project. It initializes the application and handles the main workflow.
- **evaluate_utils.py**: Includes functions for evaluating the performance of the model.
- **preprocessor.py**: Handles data loading, preprocessing tasks, and performs the first layer of feature selection using univariate logistic regressions with FDR correction.
- **SVSS.py**: Performs SVSS and extracts the top features. Also includes functions to save and load traces.
- **models.py**: Defines the model architecture and training procedures for Bayesian and Lasso logistic regression.
- **config.py**: Stores configuration settings and parameters used throughout the project.


### Folders
- **saved**: Contains saved traces from the Bayesian models, as well as a text file of significant features from the first layer of feature selection.
- **data**: Contains two files, `Features_Train` and `Features_Test`. For this study, data from both files are concatenated to obtain the full dataset.

## Results
For detailed results and analysis, please refer to the accompanying paper.

## Visualizations
Plots and visualizations are saved in the `images` folder.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
If you have any questions or feedback, feel free to reach out to me:\
ilam7@gatech.edu