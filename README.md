# Logistic Regression - Binary Classification Assignment

## 👨‍💻 Student Name
Rauf Saeed

## 📚 Assignment Details
This assignment implements **Logistic Regression** for **Binary Classification** on two different types of datasets:
1. **Tabular Data** (Bank Marketing Dataset)
2. **Image Data** (Flower Recognition Dataset)

---

## 🎯 What is Logistic Regression?
Logistic Regression is a machine learning algorithm used for **binary classification** (predicting Yes/No or 0/1). It uses a sigmoid function to convert predictions into probabilities between 0 and 1.

---

## 📊 Project 1: Bank Marketing (Tabular Data)

### Problem Statement
Predict whether a customer will subscribe to a **term deposit** (Yes/No)

### Dataset
- Source: UCI Machine Learning Repository
- Samples: 45,211 customer records
- Features: Age, Job, Balance, Duration, Previous contacts, etc.

### Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 87.4% |
| Precision | 90% (for Subscribe class) |
| Recall | 95% (for Subscribe class) |

### Key Features (Most Important)
1. Duration (call duration)
2. Balance (account balance)
3. Previous contacts
4. Age


---

## 🌸 Project 2: Flower Recognition (Image Data)

### Problem Statement
Classify flower images into **Daisy** or **Tulip**

### Dataset
- Source: CIFAR-10 (converted to binary classification)
- Training images: 5,000
- Image size: 64x64 pixels

### Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 100% |
| Precision | 1.00 |
| Recall | 1.00 |

### How it works
- Images are flattened from 64x64x3 (12,288 pixels) to 1D array
- Logistic Regression layer with sigmoid activation
- Output: 0 = Daisy, 1 = Tulip

---

## 🖥️ Flask Web Application

### Features
- **Home Page**: Two cards for both projects
- **Bank Marketing Page**: Form to input customer details
- **Flower Recognition Page**: Image upload functionality

### Routes Created
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home page |
| `/bank` | GET | Bank marketing form |
| `/flower` | GET | Flower upload page |
| `/predict_bank` | POST | Predict term deposit |
| `/predict_flower` | POST | Predict flower type |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Main programming language |
| Scikit-learn | Logistic Regression model (tabular data) |
| TensorFlow/Keras | Logistic Regression model (image data) |
| Flask | Web framework |
| HTML/CSS/JS | Frontend interface |
| Joblib | Save/load sklearn models |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Data visualization |

---

## 📁 Project Structure
