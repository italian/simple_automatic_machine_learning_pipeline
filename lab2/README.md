[![ru](https://img.shields.io/badge/lang-ru-red.svg)](README.ru.md)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/italian/simple_automatic_machine_learning_pipeline/blob/main/LICENSE)

# Simple automated machine learning pipeline on Jenkins

This is an implementation of a simple pipeline for automating the operation of a machine learning model on Jenkins. The pipeline consists of several key stages described in the file [Jenkinsfile](./Jenkinsfile).

This pipeline uses the `wine` dataset, accessible via the `load_wine` function from the `scikit-learn` library. This is a classic and very simple dataset for multiclass classification tasks. The dataset contains 13 different parameters for wines with 178 samples. Various machine learning approaches, including classification, can be used to work with this dataset to predict the class of wine based on its features.

## Stages

### 1. Setting up a Python environment

This stage creates the Python virtual environment, if it is not already created, activates it and installs dependencies from the file [requirements.txt](./requirements.txt).

### 2. Creating a dataset

The [create_dataset.py](./python_scripts/create_dataset.py) script generates a wine dataset using the `load_wine` dataset from the `sklearn.datasets` library, and saves it in `CSV` format in the `data` folder.

### 3. Data preprocessing

The [data_preprocessing.py](./python_scripts/data_preprocessing.py) script performs data preprocessing, including selecting the top five features using `SelectKBest` and the `chi2` method, and standardising the data using `StandardScaler`. The data are divided into training and test samples.

### 4. Training of the model

The [train_model.py](./python_scripts/train_model.py) script creates and trains a random forest model on training data. After training, the model is saved in `.pkl` format for future use.

### 5. Model testing

The [test_model.py](./python_scripts/test_model.py) script benchmarks the performance of a trained model on test data. The metrics `accuracy`, `precision`, `recall`, `F1-score` are used for performance evaluation.

## Usage

To run the pipeline, configure Jenkins to execute `Jenkinsfile`:
1. **Creating a new Pipeline project**.
   - In Jenkins, select New Item in the upper left corner.
   - Enter a name for your project, such as lab2.
   - ВSelect Pipeline.
   - Click OK.
2. **Setting up the Pipeline**.
   - Under Pipeline, select Pipeline script from SCM.
   - In the Repository URL field, enter the path to this repository (`https://github.com/italian/simple_automatic_machine_learning_pipeline`).
   - Make sure the correct main branch is selected.
   - In the Script Path field, specify the path to Jenkinsfile - `lab2/Jenkinsfile`.
   - Click Save.
3. **Starting Pipeline**.
   - Start Pipeline by clicking on Build Now in the left side menu of the project in Jenkins.

## Requirements

- Python 3.7+
- Libraries:
    - pandas==1.3.3
    - scikit-learn==1.4.1.post1

## Analysing the quality of the model

As a result of training and testing the model with this pipeline, we obtained metrics showing accuracy, completeness, F1-estimation and precision equal to 1.0 on both training and test data. These indicate perfect performance of the model. However, one should be careful in interpreting such results as they may indicate several possible situations:

- **Overfitting**. The model may be too complex and "memorise" the training data, including noise and emissions.
- **A simple task or a small data set**. If the task is simple or the dataset is small, the model can easily achieve high accuracy without significant risk of overfitting.
- **Class balance**. If the classes in the data are balanced, the model can show high accuracy even if it is not perfectly good.
- **Error in estimation**. There may have been an error in the model estimation process, such as using the wrong metrics or not separating the data correctly.

## Recommendations for improvement

### **Using other machine learning models**

You can try using other models such as Gradient Boosting, SVM or neural networks. This can help to compare the quality of the models and choose the most suitable one for a particular task. For example, Gradient Boosting may perform better on some tasks due to its ability to handle non-linear dependencies and robustness to outliers. SVM may be preferred for tasks with high feature dimensionality, while neural networks may be effective for tasks with a large number of features and complex data structure.

### **Applying cross-validation**

It is a good idea to apply cross-validation, which provides a more robust estimate of model quality by splitting the data into multiple subsets and training the model on different combinations of these subsets. This helps to reduce the impact of data partitioning randomness on model estimation and gives a more accurate representation of model performance.

### **Improvement of the pre-processing stage**

In the preprocessing stage, we select the k=5 best features using SelectKBest and the chi2 method. This approach can be improved by usage of other feature selection methods, such as Recursive Feature Elimination (RFE), which can take into account the relationships between features and more accurately determine the importance of features to the classification task.

### **Optimisation of `n_estimators` parameter in `RandomForestClassifier`**

The choice of the optimal value of `n_estimators` depends on the particular problem and data. One approach is to use cross-validation to evaluate the performance of the model with different values of `n_estimators` and select the value at which the model performs best. Another approach is to use methods such as `RandomisedSearchCV` from `scikit-learn` to automatically find the optimal parameter value.

It is important to note that although increasing `n_estimators` can improve the performance of the model, there is a limit after which additional trees will not bring significant improvement. This is because the random forest tends to stabilise after a certain number of trees is reached, and adding more trees will not result in additional improvement.

## Conclusions

It is recommended to perform additional in-depth analysis of the model quality using cross-validation and performance comparison with other machine learning models. It is also worth considering improving the data preprocessing stage and optimising the `n_estimators` parameter in the `RandomForestClassifier` to improve model performance.

## License

This project is distributed under the MIT licence.