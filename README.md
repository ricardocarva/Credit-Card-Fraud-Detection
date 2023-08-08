# CAP4770-Credit-Card-Fraud-Detection

## Synopsis
This project focuses on creating a robust fraud detection scheme for credit card transactions, utilizing the
data science techniques learned during the Introduction to Data Science course at the University of
Florida. Our objective is to utilize the predictive capabilities of Random Forest as a powerful model to
accurately identify fraudulent transactions. Through the training and optimization of our model, we aim
to mitigate financial losses for businesses and customers alike, ensuring their protection and
safeguarding their interests.

## Dataset
We will be utilizing the Credit Card Fraud Detection dataset obtained from Kaggle. This dataset contains
transactions made by European customers over a two-day period in September 2013. It consists of
284,807 transactions, with 492 identified as fraudulent. Additionally, this dataset presents a class
imbalance issue. The dataset primarily consists of numerical values resulting from a PCA transformation
(for customer privacy), with the exception of 'Time' and 'Amount.' Although the specific features are
unknown, our analysis will focus on determining the most significant numerical features.

## Dataset URL
Credit Card Fraud Detection https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Problem Statement
Numerous global transactions are conducted through credit cards, but determining the authenticity of a
transaction poses a challenge. The task at hand is to effectively process and analyze these transactions in
real-time, aiming to distinguish between fraudulent and legitimate ones. This endeavor is highly intricate, if not unattainable, without leveraging the computational power of computers to undertake this task on our behalf, especially considering the imbalances in data distribution. Our objective is to
develop a machine learning-based fraud detection system that can successfully address this classification issue.

## Evaluation Metrics
The two evaluation metrics we will focus on are F-1 score and Area Under the Precision-Recall Curve
(AUPRC). These techniques are of use to us specifically because they account for imbalances in the
dataset. Fraudulent transactions can often exhibit complex and non-linear patterns which make certain
metrics like the confusion matrix less desirable. The F1 score will be a suitable choice when you want a
balanced evaluation metric that considers both precision and recall and it handles imbalanced class
distributions. AUPRC is useful when the focus is on the performance of the minority class, especially in
imbalance datasets. It provides a comprehensive evaluation by considering the precision and recall
trade-off at various threshold settings. If we find that these two metrics are not enough, we will
incorporate other evaluation metrics and document it in our final project report.

## Baseline Techniques
The baseline techniques used for a random forest classification in our project’s case involves the
following steps.
| #  |  Step  |  Description  |
| :---------------- | :------: | ----: |
| 1  | Data Preparation   |  Luckily for us the data has been well cleaned so this step is largely done for us. However, we can split and manipulate the data in various ways if necessary as long as we ensure the class imbalance is maintained in both sets. |
| 2  |  Model   |  Training Initially, we will train with default parameters and fine tune the parameters to achieve a higher degree of accuracy when the model makes predictions. |
| 3  |  Evaluation   |  We will calculate the F1 score and AUPRC to assess the performance of the model. |
| 4  |  Comparison   |  Once the results are obtained we will compare the metrics with the desired benchmarks and previous results. |
| 5  |  Model Improvement   |  Based on the results we have obtained, we will fine tune the model and re-evaluate. |

## Environment Setup
### Python Environment: 
Ensure you have Python installed (preferably version 3.9) on your system.
### Jupyter Notebook: 
This project is intended to be run using Jupyter Notebook. If you don't have Jupyter Notebook installed, you can install it using the following command: `pip install jupyter`
### Package Installation: 
Install the required packages by running the following command in your terminal or command prompt: `pip install numpy pandas matplotlib seaborn plotly scikit-learn imbalanced-learn Pillow plotly_express`

### Running Jupyter Notebook in PyCharm
1. Install PyCharm:
If you haven't already, download and install PyCharm from the official website: [PyCharm Download]([url](https://www.jetbrains.com/pycharm/download/?section=windows)).

3. Open Project: Open your project in PyCharm.

4. Create or Open Jupyter Notebook:
   - If you have an existing Jupyter Notebook file (with .ipynb extension), you can directly open it in PyCharm.
   - If you don't have an existing Jupyter Notebook, you can create a new one by right-clicking on the project folder in the Project Explorer, selecting "New" > "Python File," and giving it a .ipynb extension.

5. Activate Virtual Environment (Optional):
   - If you're using a virtual environment, make sure to activate it using the PyCharm terminal. This ensures that the Jupyter Notebook runs in the correct environment.
   - To activate the virtual environment:
`source path_to_your_virtual_environment/bin/activate`

6. Run Jupyter Notebook:
   - Open the Jupyter Notebook file in the PyCharm editor.
   - You can run each cell by clicking on it and then clicking the "Run" button in the cell toolbar or using the keyboard shortcut Shift + Enter.
   - Alternatively, you can run all cells by selecting "Run" > "Run All Cells" from the Jupyter Notebook menu.

### Code Execution Steps
1. Importing Libraries:
   - The code begins by importing necessary libraries for data analysis and visualization.

2. Loading Dataset:
   - The dataset is loaded from the specified CSV file path using the pd.read_csv() function.

3. Data Exploration and Visualization:
   - The shape of the dataset is printed using data.shape.
   - A pie chart is created using Plotly Express (px.pie()) to visualize the distribution of fraud vs. valid transactions.
   - Fraudulent and valid transactions are separated and their counts are displayed.
   - Descriptive statistics for the 'Amount' column are computed and printed for both fraudulent and valid transactions.
   - A correlation matrix heatmap is created using Seaborn to visualize the correlation between features.

4. Data Preprocessing:
   - The target variable 'Class' is separated from the features to create X (features) and Y (target) arrays.
   - The dataset is split into training and testing sets using train_test_split() from scikit-learn.

5. Handling Class Imbalance:
   - Different resampling methods are defined in the sample_methods dictionary, including SMOTE, NearMiss, RandomOverSampler, and RandomUnderSampler.
   - The split_features() function separates features and target, and the rebalance_class() function applies different resampling methods and displays the shape - and balance of the resulting datasets.
   - The training set is rebalanced using the defined resampling methods.

6. Random Forest Classifier:
   - The RandomForestClassifier from scikit-learn is imported.
   - An instance of the classifier is created with a specified number of estimators.
   - The model is trained using the rebalanced training data.

7. Prediction and Evaluation:
   - The model's predictions are obtained using predict() on the test data.
   - Various evaluation metrics (accuracy, precision, recall, F1 score, AUPRC) are computed using scikit-learn's metrics functions.
   - The evaluation results are displayed and visualized using a bar chart.

8. Precision-Recall Curve:
   - The precision-recall curve and display are created using scikit-learn's precision_recall_curve() and PrecisionRecallDisplay() functions.

### Note
Make sure to replace the CSV file path with the correct path to your "creditcard.csv" dataset file.
The code involves data visualization and analysis, handling class imbalance, and training and evaluating a Random Forest Classifier for fraud detection.

## References
[1] Kiran Deep Singh, P. Singh, and Sandeep Singh Kang, “Ensembled-based credit card fraud detection in
online transactions,” Jan. 2022, doi: https://doi.org/10.1063/5.0108873.

[2] C. Whitrow, D. J. Hand, P. Juszczak, D. Weston, and N. M. Adams, “Transaction aggregation as a
strategy for credit card fraud detection,” Data Mining and Knowledge Discovery, vol. 18, no. 1, pp. 30–55,
Jul. 2008, doi: https://doi.org/10.1007/s10618-008-0116-z.
