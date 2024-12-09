## PREDICTING CUSTOMER CHURN FOR ENHANCED RETENTION STRATEGIES AT SYRIATEL TELECOMUNICATION
### by Michelle Anyango
![image link](https://github.com/Michdev2024/Moringa-phase-3-project/blob/main/photo%201.jpg)


## Overview

This project focuses on predicting customer churn for Syriatel, a leading telecommunication provider in Syria. **Customer churn**—when customers stop using a service—is a critical issue for the telecom industry. Understanding and addressing the factors that lead to churn can significantly improve customer retention and drive revenue growth.

By leveraging machine learning, this project aims to develop a reliable churn prediction model and uncover actionable insights to guide Syriatel in its customer retention strategies.

---
![image link](https://github.com/Michdev2024/Moringa-phase-3-project/blob/main/photo%202.jpg)

## Business Problem

To grow their revenue, telecommunication companies must attract new customers while retaining existing ones. Customer churn impacts the bottom line, often driven by factors like:
- Better pricing from competitors
- Poor service quality
- Lack of engagement

Recognizing that retaining existing customers is more cost-effective than acquiring new ones, Syriatel aims to:
1. **Predict customer churn** using historical data.
2. **Identify factors driving churn**, enabling actionable strategies to reduce it.

---

## Data Understanding

The dataset used in this project was sourced from [Kaggle's Churn in Telecoms Dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset). 

### Dataset Highlights:
- **Rows (Customers)**: 3,333
- **Columns (Features)**: 21

### Feature Categories:
1. **Customer Information**: Basic details like demographics.
2. **Account Information**: Subscription and billing details.
3. **Usage Metrics**: Data on calls, minutes, and other usage behaviors.
4. **Customer Service Interaction**: Records of customer service calls.
5. **Target Variable**: Binary flag indicating whether a customer churned.

---

## Objectives

1. **Build a Machine Learning Model**: Develop and evaluate models to accurately predict customer churn.
2. **Identify Key Features**: Pinpoint the most influential factors driving churn.

---
## Loading and cleaning data 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier

---
## Approach

1. **Exploratory Data Analysis (EDA)**: Understand trends, distributions, and relationships in the data.
2. **Feature Engineering**: Prepare and optimize features for modeling.
3. **Model Building**: Experiment with various machine learning models, including:
   - Baseline model: Logistic Regression
   - Advanced models: Decision Trees, Random Forest
4. **Model Tuning**: Optimize hyperparameters for improved performance.
5. **Evaluation**: Assess model performance using metrics like accuracy and feature importance.

---

## Results and Key Insights

1. **Best Model**: 
   - The **Random Forest** model achieved the highest predictive performance, outperforming Logistic Regression and Decision Trees.

2. **Key Predictive Features**:
   - **Total Day Minutes**
   - **Customer Service Calls**
   - **Subscription to International Plans**

3. **Significance**: These features provide actionable insights to design targeted retention strategies for at-risk customers.

---

## Visualization 
![image link](https://github.com/Michdev2024/Moringa-phase-3-project/blob/main/photo%204.png)
Distribution of Numeric feature in the data set 

---

![image link](https://github.com/Michdev2024/Moringa-phase-3-project/blob/main/photo%206.png)
 The bar graph shows the distribution of total minutes across various call categories "total eve minutes" and "total eve minutes"  being the two highest and "total intl minutes" being the lowest  

 ---
 
![image link](https://github.com/Michdev2024/Moringa-phase-3-project/blob/main/photo%207.png)
The heatmap shows correlation of the different elements 

---
## Conclusion

The churn prediction analysis conducted for SyriaTel aimed to build a reliable classifier to identify customers at risk of terminating their services. Through extensive data exploration, preparation, and modeling, several key insights emerged:

- **Model Performance**: Random Forest demonstrated the highest effectiveness for churn prediction, outperforming Logistic Regression and Decision Trees. Its superior accuracy and predictive power make it the ideal choice for SyriaTel's churn prediction system.

- **Key Predictive Features**: Total day minutes, customer service calls, and subscription to the international plan were identified as the most significant indicators of churn. These insights can help SyriaTel design proactive retention strategies focused on high-risk customers.


This project demonstrates how machine learning can empower Syriatel to predict churn and implement data-driven retention strategies. By prioritizing the identified key features and acting on the recommendations, Syriatel can significantly reduce customer churn and foster long-term relationships.


---

## Recommendations

1. **Enhance Customer Experience**:
   - Improve call quality by investing in advanced infrastructure and technology.
   - Streamline customer service operations to resolve issues promptly.

2. **Design Tailored Plans**:
   - Develop attractive international plans to retain subscribers with high international usage.

3. **Implement Proactive Retention Strategies**:
   - Offer personalized promotions and loyalty rewards to high-risk customers.

4. **Continuously Monitor Trends**:
   - Regularly update models and churn analysis to stay aligned with customer behavior and market dynamics.


---
![image link](https://github.com/Michdev2024/Moringa-phase-3-project/blob/main/photo%203.jpg)
