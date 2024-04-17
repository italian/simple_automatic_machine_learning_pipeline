[![ru](https://img.shields.io/badge/lang-ru-red.svg)](README.ru.md)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/italian/simple_automatic_machine_learning_pipeline/blob/main/LICENSE)

# Simple automatic machine learning pipeline using bash

This is an implementation of a simple pipeline for automating machine learning model work in bash.

Developed as part of a training task and includes several key stages.

## Stages

### 1. Data Creation

The script [data_creation.py](python_scripts/data_creation.py) generates various temperature data sets, including anomalies and noise, and stores them in the `train` and `test` folders. Data is generated using the `numpy` library and saved in `CSV` format.

### 2. Data preprocessing

The [model_preprocessing.py](python_scripts/model_preprocessing.py) script performs data standardisation using `sklearn.preprocessing.StandardScaler`. This is necessary to ensure stability of the model training.

### 3. Model training

The script [model_preparation.py](python_scripts/model_preparation.py) creates and trains a machine learning model using data from the `train` folder. The model used is logistic regression from the `sklearn` library. After training, the model is saved in `.pkl` format for further use.

### 4. Model testing

The [model_testing.py](python_scripts/model_testing.py) script tests the performance of a trained model on data from the `test` folder. The metrics `accuracy`, `precision`, `recall`, `F1-score` are used to evaluate the performance.

### 5. Pipeline automation

The bash script [pipeline.sh](pipeline.sh) allows you to run the entire pipeline automatically by running all python scripts sequentially. The script also creates a virtual environment and installs the necessary dependencies from the `requirements.txt` file.

## Usage

To run a pipeline, clone the repository and run the command `./pipeline.sh` in a terminal.

### Parameters:

- **Optional parameter**: Number of datasets to create. By default, one dataset is created.

Usage Examples:

- Running the script without parameters (one dataset is created):
    ```shell
    ./pipeline.sh
    ```
- Running the script with the number of datasets specified (e.g., 5 datasets):
    ```shell
    ./pipeline.sh 5
    ```

Make sure you have all the required dependencies installed, as specified in the [requirements.txt](./requirements.txt) file.

### Example output
```
Virtual environment '.venv' is activated.
All dependencies are already installed.

The model for the data set 1 is trained.
 Accuracy  Precision  Recall  F1-score
 0.950909   0.925926     0.5  0.649351
--------------------

The model for dataset 1 is tested.
 Accuracy  Precision  Recall  F1-score
 0.954545        1.0     0.5  0.666667
--------------------
```

## Requirements

- Python 3.7+
- Libraries:
    - numpy,
    - pandas,
    - sklearn

## Possible problems and solutions

- **Virtual environment activation problem in `pipeline.sh`**: The use of the `$VIRTUAL_ENV` variable in the [pipeline.sh](pipeline.sh) script inside the `activate_venv` function to check for virtual environment activation may be unreliable in some cases, such as when the script is run inside an `tmux` or `screen` session created in an activated virtual environment. In such cases, `$VIRTUAL_ENV` may remain set even if the virtual environment is not active in the current context.
    - **Possible Solution**: Use other methods to verify virtual environment activation or avoid using virtual environments in such scenarios.

## Improvements

- **Add support for other machine learning models**: The current implementation only uses logistic regression. Adding support for other models, such as decision trees or neural networks, can extend the capabilities of the project.
- **Integration with CI/CD systems**: Integration with Continuous Integration and Continuous Delivery (CI/CD) systems can automate the model testing and deployment process, improving the development process.

## Licence

This project is distributed under the MIT licence.