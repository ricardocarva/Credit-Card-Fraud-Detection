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
#   Step   Description
1  | Data| Preparation   |  Luckily for us the data has been well cleaned so this step is largely done for us.
However, we can split and manipulate the data in various ways if necessary as
long as we ensure the class imbalance is maintained in both sets
2  |  Model   |  Training Initially, we will train with default parameters and fine tune the parameters to
achieve a higher degree of accuracy when the model makes predictions.
3  |  Evaluation   |  We will calculate the F1 score and AUPRC to assess the performance of the
model.
4  |  Comparison   |  Once the results are obtained we will compare the metrics with the desired
benchmarks and previous results.
5  |  Model Improvement   |  Based on the results we have obtained, we will fine tune the model and
re-evaluate.

## References
[1] Kiran Deep Singh, P. Singh, and Sandeep Singh Kang, “Ensembled-based credit card fraud detection in
online transactions,” Jan. 2022, doi: https://doi.org/10.1063/5.0108873.
[2] C. Whitrow, D. J. Hand, P. Juszczak, D. Weston, and N. M. Adams, “Transaction aggregation as a
strategy for credit card fraud detection,” Data Mining and Knowledge Discovery, vol. 18, no. 1, pp. 30–55,
Jul. 2008, doi: https://doi.org/10.1007/s10618-008-0116-z.
