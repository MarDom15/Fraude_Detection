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
The **Credit Card Fraud Detection** project leverages machine learning and deep learning techniques to identify fraudulent transactions. The model is developed to improve detection accuracy while maintaining efficiency and scalability. 

### Theoretical Context:
Fraud detection models aim to classify transactions as either legitimate or fraudulent. Machine learning techniques like Logistic Regression and Random Forest, along with deep learning (MLP), are employed to achieve this goal.

🎯 **Main Objectives:**
- Identify fraudulent transactions with high accuracy.
- Provide a scalable and interactive deployment solution.
- Implement a robust CI/CD pipeline for consistent updates.

[View Visuals for Introduction](#)

---

## 🔍 **2. Data Exploration (EDA)**
The dataset provides transaction information, including numerical attributes, time, amount, and a fraud label.

### Key Steps:
1. Analyze class distribution (fraud vs. legitimate).
2. Visualize feature correlations.
3. Check for outliers and anomalies.

### Theoretical Context:
The dataset is highly imbalanced, requiring techniques like SMOTE (Synthetic Minority Oversampling) or class weighting during model training.

📊 **Visualizations:**
- **Class Distribution:** Fraud vs. legitimate transactions.
- **Feature Correlation Heatmap:** Identifies significant relationships between features.

[View Visuals for EDA](#)

---

## 🛠️ **3. Data Preparation**
Data preparation ensures consistency and suitability for modeling.

### Key Steps:
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

