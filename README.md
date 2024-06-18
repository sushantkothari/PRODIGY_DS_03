# PRODIGY_DS_03

## Bank Marketing Campaign Analysis

This repository contains a Python script (`Task 3.ipynb`) that performs exploratory data analysis (EDA) and builds a machine learning model for predicting whether a customer will subscribe to a term deposit based on information from a bank marketing campaign.

## Description

The script follows these steps:

1. **Data Extraction**: Downloads the "Bank Marketing" dataset from the UCI Machine Learning Repository and extracts the relevant files.
2. **Data Loading**: Loads the dataset into a Pandas DataFrame.
3. **Exploratory Data Analysis (EDA)**: Conducts various analyses, including summary statistics, target variable distribution, pairplots, correlation matrices, and boxplots to understand the data better.
4. **Data Preprocessing**: Handles numerical and categorical features using scikit-learn pipelines for scaling and one-hot encoding, respectively.
5. **Train-Test Split**: Splits the data into training and testing sets.
6. **Class Imbalance Handling**: Applies the SMOTE technique to handle the class imbalance in the training data.
7. **Model Training**: Trains a Decision Tree Classifier on the oversampled training data.
8. **Cross-Validation**: Performs stratified 5-fold cross-validation on the trained model to evaluate its performance.
9. **Model Evaluation**: Evaluates the model's performance on the test set using various metrics, including accuracy, precision, recall, F1-score, ROC AUC, and average precision.
10. **Hyperparameter Tuning**: Employs GridSearchCV to tune the hyperparameters of the Decision Tree Classifier.
11. **Model Visualization**: Visualizes the trained Decision Tree model for better interpretability.

## Requirements

To run this script, you'll need the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn

You can install these libraries using pip:
```
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

```
## Usage

1. Clone this repository to your local machine.
2. Navigate to the repository directory.
3. Run the script using the following command:
```
python Task 3.ipynb
```
The script will execute, and you'll see the output and visualizations in your console or a separate window, depending on your system configuration.

## Contributing

Contributions to this project are welcome. If you have any suggestions, bug fixes, or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The "Bank Marketing" dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).
- The code utilizes various Python libraries, including pandas, numpy, matplotlib, seaborn, scikit-learn, and imbalanced-learn.

