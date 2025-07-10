# 🩺 Diabetes Prediction using Linear Regression

This project explores the use of **Linear Regression** to predict whether a person is diabetic or not, based on medical data. The dataset used comes from the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

Although classification models are more suited to binary outcomes, this experiment uses regression followed by thresholding to observe how well Linear Regression performs in a classification setting.

---

## 📌 Objectives

- Clean and preprocess medical data for modeling.
- Apply a Linear Regression model for binary prediction.
- Evaluate the model using classification metrics.
- Visualize the confusion matrix for better insight.

---

## 📚 Theoretical Background

### 🔹 Linear Regression Hypothesis Function

The hypothesis function for Linear Regression is given by:

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
$$

Where:
- <span style="font-size: 20px;">&#x03B8;<sub>i</sub></span>: model coefficients (weights)
- <span style="font-size: 20px;">x<sub>i</sub></span>: feature values
- <span style="font-size: 20px;">&#x03B8;<sub>0</sub></span>: intercept term

### 🔹 Cost Function (Mean Squared Error)

The cost function used to optimize the parameters in Linear Regression is the **Mean Squared Error (MSE)**:

$$
J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2
$$

Where:
- <span style="font-size: 20px;">m</span>: number of training samples
- <span style="font-size: 20px;">&#x03B8;<sub>i</sub></span>: predicted output for sample <span style="font-size: 20px;">i</span>
- <span style="font-size: 20px;">y<sub>i</sub></span>: actual output for sample <span style="font-size: 20px;">i</span>

---

## ⚙️ How It Works

1. **Load the dataset** using `pandas`.
2. **Replace invalid zero values** in key columns (`Glucose`, `BloodPressure`, `BMI`, etc.) with column mean.
3. **Adjust special rows** by assigning min/max glucose values.
4. **Split the data** into training and test sets.
5. **Train a Linear Regression model** using `scikit-learn`.
6. **Round predictions** to either 0 or 1.
7. **Evaluate the model** with accuracy, confusion matrix, precision, recall, and F1 score.

---

## 🧪 Evaluation Metrics

- **Accuracy**:  
  The accuracy of the model is the ratio of correct predictions to the total number of predictions:

  <p align="center">
      <span style="font-size: 20px;">Accuracy = (TP + TN) / (TP + TN + FP + FN)</span>
  </p>

- **Precision**:  
  Precision measures how many of the predicted positive cases are actually positive:

  <p align="center">
      <span style="font-size: 20px;">Precision = TP / (TP + FP)</span>
  </p>

- **Recall**:  
  Recall measures how many of the actual positive cases were correctly predicted:

  <p align="center">
      <span style="font-size: 20px;">Recall = TP / (TP + FN)</span>
  </p>

- **F1 Score**:  
  The F1 score is the harmonic mean of precision and recall:

  <p align="center">
      <span style="font-size: 20px;">F1 = 2 × (Precision × Recall) / (Precision + Recall)</span>
  </p>

Where:
- <span style="font-size: 20px;">TP</span>: True Positive
- <span style="font-size: 20px;">FP</span>: False Positive
- <span style="font-size: 20px;">TN</span>: True Negative
- <span style="font-size: 20px;">FN</span>: False Negative

---

## 📊 Results

- **Accuracy**: 77.92%
- **Precision**: 0.72
- **Recall**: 0.62
- **F1 Score**: 0.67

---

## 📁 Files Included

- `Diabetes_Prediction.ipynb`: Jupyter Notebook with complete code and output.
- `diabetes.csv`: Dataset file (must be placed in the correct path if running locally).
- `README.md`: Project documentation.

---

## 🚀 Requirements

Install required packages using:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
