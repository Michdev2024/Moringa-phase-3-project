
# Syriatel Customer Churn Prediction

## Overview

This project focuses on predicting customer churn for Syriatel, a leading telecommunication provider in Syria. **Customer churn**—when customers stop using a service—is a critical issue for the telecom industry. Understanding and addressing the factors that lead to churn can significantly improve customer retention and drive revenue growth.

By leveraging machine learning, this project aims to develop a reliable churn prediction model and uncover actionable insights to guide Syriatel in its customer retention strategies.

---

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

## Conclusion

This project demonstrates how machine learning can empower Syriatel to predict churn and implement data-driven retention strategies. By prioritizing the identified key features and acting on the recommendations, Syriatel can significantly reduce customer churn and foster long-term relationships.

---

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/syriatel-churn-prediction.git
