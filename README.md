# 🌟 **Credit Card Fraud Detection**
*Fraud detection using Logistic Regression, Random Forest, and Deep Learning (MLP) with MLOps and Streamlit.*

---

## 🗂️ **Table of Contents**
1. [✨ Introduction](#introduction)  
2. [🔍 Data Exploration (EDA)](#data-exploration-eda)  
3. [🛠️ Data Preparation](#data-preparation)  
4. [🏃️ Model Training](#model-training)  
5. [✅ Testing and Evaluation](#testing-and-evaluation)  
6. [🚀 Deployment](#deployment)  
    - [📦 Creating a Docker Image](#creating-a-docker-image)  
    - [🌐 Deployment on Streamlit and AWS](#deployment-on-streamlit-and-aws)  
7. [📊 Monitoring and MLOps](#monitoring-and-mlops)  
    - [🔄 Building an MLOps Pipeline](#building-an-mlops-pipeline)  
8. [🗂 Data](#data)  
9. [🙌 Contributors](#contributors)  

---

## ✨ **1. Introduction**
The **Credit Card Fraud Detection**This project focuses on detecting fraudulent and non-fraudulent banking transactions by leveraging machine learning and deep learning techniques. The algorithms employed include logistic regression, multilayer perceptrons (MLP), and random forests, all trained to maximize accuracy and uncover complex patterns in transactional data. The pipeline has been deployed on an AWS EC2 instance, ensuring scalability and efficient handling of large-scale computations. Feature engineering played a crucial role in extracting relevant indicators and identifying anomalous behaviors. This project emphasizes a balance between performance, accuracy, and adaptability to modern financial systems.   

### Theoretical Context:
Fraud detection models aim to classify transactions as either fraudulent or non-fraudulent. To achieve this goal, machine learning techniques such as Logistic Regression and Random Forest are used, alongside deep learning methods like Multilayer Perceptrons (MLP). These algorithms enable the extraction of complex patterns in transactional data, thereby enhancing the accuracy and reliability of predictions.

🎯 **Main Objectives:**
- Identify fraudulent transactions with high accuracy.
- Provide a scalable and interactive deployment solution.
- Implement a robust CI/CD pipeline for consistent updates.

---

## 🔍 **2. Data Exploration (EDA)**
The dataset provides transaction information, including numerical attributes, amount, and a fraud label.
Step 1: Load the Dataset

-  Data set: Head of Dataset:
  ![Head of Data set](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/Head_data.png)

Step 1': Check for missing values in the DataFrame

-  MIssing Values:Check for missing values in the DataFrame :
  ![Check for missing values in the DataFrame](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/missing_values.png)

### Key Steps:
Step 2: Display DataFrame Information
In this step, we display the DataFrame information, including the number of rows, the number of columns, the data type of each column, and whether any missing values are present. This helps in understanding the structure of the data and identifying potential issues.

-  Data set: Dataframe Information:
  ![Dataframe Information](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/data_types.png)

Step 3: Visualize the distribution of classes
In this step, we visualize the distribution of classes (fraudulent and non-fraudulent) in the dataset. This helps in understanding the class balance and identifying any potential imbalances, which can affect the performance of machine learning models.

-  Distribution: Visualisation of Distribution of classes:
  ![Distribution of classes](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/class_distribution.png)

Step 4: Statistical summary of the data
In this step, we generate a statistical summary of the data by calculating measures such as mean, standard deviation, minimum and maximum values, and quartiles. This provides an overview of the data's characteristics and helps in detecting any anomalies or outliers in the features.

-  Statistical Summary: Statistical Summary of the data:
  ![Statistical Summary](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/statistical_data_sunnary.png)

Step 5: Visualize the distribution of fraudulent transaction amounts
In this step, we visualize the distribution of fraudulent transaction amounts. This helps in understanding the patterns related to fraudulent transactions and identifying specific trends, such as higher or lower amounts associated with fraud.

-  Visualisation of distribution: Visualize the distribution of fraudulent transaction amounts :
  ![Visualize the distribution of fraudulent transaction amounts](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/distribution_fraudulent_transaction.png)

Step 6: Visualize the distribution of non-fraudulent transaction amounts
In this step, we visualize the distribution of non-fraudulent transaction amounts. This analysis helps in understanding the patterns related to legitimate transactions and identifying trends or specific characteristics of amounts associated with non-fraudulent transactions.

-  Visualisation of distribution: Visualize the distribution of non-fraudulent transaction amounts :
  ![Visualize the distribution of non-fraudulent transaction amounts](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/distribution_nonfraudulent_transaction.png)

Step 7: Calculation of the correlation of all data with the target label 'Class'.

Interpretation of correlation results:
- Perfect positive correlation (1):
When one variable increases, the other increases in a proportional manner.
This means there is a direct relationship between the two variables. For instance, a correlation of 1 between two variables implies that for every unit increase in the first variable, the second variable increases in a determined way.
- No correlation (0):
The variables are not related to each other.
This indicates that there is no linear relationship between the two variables, meaning that an increase in one does not have a direct or predictable impact on the other.
- Perfect negative correlation (-1):
When one variable increases, the other decreases in a proportional manner.
This implies a strict inverse relationship between the two variables. For example, a correlation of -1 means that for every unit increase in the first variable, the second variable will decrease in a defined way.
Strong and weak correlations:
Strongly positively correlated variables:

- V4 (0.735981) and V11 (0.724278) show significant positive relationships with the target variable Class. These variables have correlations close to 1, indicating they are strong predictors for Class in a machine learning model. They increase together significantly.
Weakly positively correlated variables:

- V2 (0.491878) and Amount (0.002261) show positive relationships, but their strength is weaker. Although there is a tendency for both to increase together, the impact is not as strong, and their contribution to predicting Class is lesser.
Weakly negatively correlated variables:

- V15 (-0.037948) to V14 (-0.805669) indicate negative relationships. For instance, V12 (-0.768579) and V14 (-0.805669) have strong negative correlations. This means that as these variables increase, the likelihood of Class being fraudulent decreases, which could be critical for fraud detection.

Practical applications:
The most important variables for prediction are those with values close to 1 or -1, as they are strongly correlated with Class. Variables with correlations close to 0, like V22 or Amount, might be less relevant in a model unless they hold specific contextual significance.

C/C
Strong positive variables: V4, V11
Strong negative variables: V12, V14, V3
Variables close to 0: V22, Amount

-  Correlation whit classe: table for calculating the % correlation with the target value  :
  ![table for calculating the % correlation with the target value](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/Correlationcal.png)

Step 8: Visualize the correlation matrix
In this step, we visualize the correlation matrix as a heatmap to observe the relationships between all the variables in the dataset. The heatmap allows for a quick visualization of the linear associations between variables, whether they are positive, negative, or zero. The colors in the heatmap represent the strength of the correlation: a warm color (like red) indicates a strong positive correlation, while a cool color (like blue) indicates a strong negative correlation. This makes it easier to identify the most influential variables for predictive models.

-  Correlation Heatmap: Correlation Heatmap :
  ![Correlation Heatmap](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/heatmap_correlation_different_variables.png)

-  Correlation Heatmap: Correlation Heatmap for fraudulent transactios  :
  ![Correlation Heatmap for Fraud Transaction](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/heatmap_correlation_fraudulent_transactions.png) 

-  Correlation Heatmap: Correlation Heatmap for non-fraudulent transactios  :
  ![Correlation Heatmap for non-Fraud Transaction](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/heatmap_correlation_nonfraudulent_transaction.png) 


Step 9: Visualize the distribution of transaction amounts with a boxplot
In this step, we use a boxplot to visualize the distribution of transaction amounts. A boxplot provides a summary of the distribution of numerical values by showing the quartiles, median, and potential outliers. This helps in understanding the spread of transaction amounts and detecting extreme values that could influence our predictive models.

-  Distribution of transaction amounts: Visualize the distribution of transaction amounts with a boxplot :
  ![Visualize the distribution of transaction amounts with a boxplot](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/distribution_of_transaction_amounts.png)

Step 10: Visualize the distribution of principal features (PCA components)
In this step, we use PCA (Principal Component Analysis) to reduce the dimensionality of the data and visualize the main features or components extracted. We display the distribution of the principal components to better understand how the variables are represented in a lower-dimensional space. This helps in analyzing whether certain components explain a significant portion of the data variance and in detecting underlying patterns that can be useful for classification or fraud detection.

-  Feature Distribution (PCA Components): Visualize the distribution of principal features :
  ![Visualize the distribution of principal features](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/feature_distribution_of_importantfeatur.png)

Step 11: Visualize relationships between features for fraudulent and non-fraudulent transactions
In this step, we visualize the relationships between features specifically for fraudulent and non-fraudulent transactions. This helps to examine how different variables interact in the context of fraudulent and non-fraudulent transactions and to identify patterns or common behaviors. By using graphs such as scatter plots, pair plots, or correlation matrices, we can better understand the complex relationships that could be leveraged for fraud detection.

-  Visualize relationships between features : Visualize relationships between features for fraudulent transactions  :
  ![Visualize relationships between features for fraudulent transactions](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/pairplot_1000_fraudulent.png)

-  Visualize relationships between features : Visualize relationships between features for non-fraudulent transactions  :
  ![Visualize relationships between features for non-fraudulent transactions](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/pairplot_1000_nonfraudulent.png)


Step 12: Visualize fraud vs non-fraud transactions with a scatterplot
In this step, we use a scatterplot to compare fraudulent and non-fraudulent transactions. This helps visualize the differences between the two classes based on two or more features. The scatterplot makes it easier to identify clusters, anomalies, or specific patterns associated with each transaction type. By using distinct colors for fraudulent and non-fraudulent transactions, this visualization highlights any potential segregation of the classes in the feature space.

-  Visualisation of fraud vs non-fraud transaction : Visualize fraud vs non-fraud transactions with a scatterplo :
  ![Visualize fraud vs non-fraud transactions with a scatterplo](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/scatterplot_id_vs_amount_fraud_Vs_nonfraud.png)


---

## 🛠️ **3. Data Preprocessing**
Data preprocessing is a critical step in machine learning projects, ensuring raw data is transformed into a suitable format for model training. In this project, preprocessing was simplified as all data is numerical, eliminating the need for encoding categorical variables. Additionally, the dataset is balanced, removing the necessity for resampling techniques like oversampling or undersampling to address class imbalances. Furthermore, important features have already been identified based on their correlation with the target variable ("class"), allowing the focus to remain on refining the model’s performance without unnecessary dimensionality reduction or feature selection steps.

### Key Steps:
Step 1 : Data Cleaning
Data cleaning is the first and most essential step in preprocessing. It involves handling missing values, duplicates, and inconsistencies in the dataset to ensure its quality and reliability. For this dataset, as the data is already numerical and complete, the cleaning process focuses on detecting and removing duplicates, handling outliers where necessary, and ensuring that all features are correctly scaled or standardized if required. A clean dataset ensures that the machine learning models can train effectively without being biased by noise or errors

- Outliers Cheking for V4, V11, V12, V14, V3 : Outlier cheking for important values :
  ![OUtliers Cheking](https://github.com/MarDom15/Fraude_Detection/blob/main/image_prog/outliers_im_importantv.png)

Step 2 : Scaling the Data (Normalization or Standardization)
Scaling the data is a crucial preprocessing step to ensure that all features contribute equally to the model's performance. Depending on the model and the nature of the dataset, we can use normalization (scaling values to a range, typically [0, 1]) or standardization (centering data around a mean of 0 with a standard deviation of 1). For this project, scaling ensures that features such as transaction amounts, which may have large ranges, do not dominate the learning process. Proper scaling improves convergence during training and enhances model performance, especially for distance-based or gradient-sensitive algorithms.

- Normalization or Standardization : Normalization or Standardization of important values :
  ![Normalization or Standardization of important values]()




1. **Feature Scaling:** Normalize numerical features for better performance in algorithms.
2. **Imbalance Handling:** Use SMOTE or similar techniques.
3. **Train-Test Split:** Divide data into training, validation, and test sets.

### Theoretical Context:
Preprocessing improves convergence and ensures that models generalize well to unseen data.

[View Visuals for Data Preparation](#)

---

## 🏃️ **4. Model Training**
The project employs three different algorithms:

1. **Logistic Regression:** A baseline model for linear separability.
2. **Random Forest:** A robust ensemble model for feature importance and high accuracy.
3. **Deep Learning (MLP):** A neural network-based approach to capture complex patterns.

### Training Steps:
- Apply cross-validation for model tuning.
- Hyperparameter optimization using grid search or random search.

### Theoretical Context:
- **Logistic Regression:** Well-suited for simple datasets with linear relationships.
- **Random Forest:** Captures non-linearities and ranks feature importance.
- **MLP:** Exploits the power of deep learning for complex patterns.

[View Visuals for Model Training](#)

---

## ✅ **5. Testing and Evaluation**
Evaluation metrics help measure model performance on unseen data.

### Metrics Used:
1. **Accuracy:** Overall correctness.
2. **Precision:** Correctness of fraud predictions.
3. **Recall:** Sensitivity to fraud cases.
4. **F1-score:** Harmonic mean of precision and recall.

🔢 **Confusion Matrix:** Provides insights into classification errors.

[View Visuals for Testing and Evaluation](#)

---

## 🚀 **6. Deployment**
The deployment involves Streamlit Community Cloud and AWS EC2, ensuring scalability and accessibility.

### 📦 **Creating a Docker Image**
A Docker image packages the application and dependencies for portability.

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy application files
COPY . /app
WORKDIR /app

# Start Streamlit
CMD ["streamlit", "run", "app.py"]
```
---

## 🌐 **Deployment on Streamlit and AWS**
1. **Streamlit Community:** Simple and fast deployment for initial testing.
2. **AWS EC2:** Robust cloud deployment using Docker and Jenkins for CI/CD.

[View Visuals for Deployment](#)

---

## 📊 **7. Monitoring and MLOps**

### Monitoring Tools:
- **Grafana:** Visualize metrics such as latency and errors.
- **Prometheus:** Collect real-time metrics.

### Building an MLOps Pipeline:
1. **CI/CD with Jenkins:** Automates code integration and deployment.
2. **Docker Hub:** Stores Docker images for easy access.

---

**Pipeline Example:**
```yaml
pipeline {
    agent any
    stages {
        stage('Build') {
            steps {
                sh 'docker build -t fraud-detection .'
            }
        }
        stage('Push to Docker Hub') {
            steps {
                sh 'docker push my-docker-hub-repo/fraud-detection'
            }
        }
        stage('Deploy to AWS') {
            steps {
                sh 'ssh ec2-user@<AWS_IP> docker run -d -p 8501:8501 my-docker-hub-repo/fraud-detection'
            }
        }
    }
}
```
---

## 🗂 **8. Data**
The dataset, available on Kaggle, includes anonymized credit card transaction data.

### Dataset Link:
- [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

The dataset contains various features related to credit card transactions, such as transaction amount, user details (anonymized), and whether the transaction was fraudulent or not. It is designed for use in machine learning models for fraud detection.

[View Visuals for Data](#)

---

## 🙌 **9. Contributors**
This project was created by:  
- **M. Domche**  
For questions or suggestions, contact me at [mdomche@example.com](mailto:mdomche@example.com).

---

