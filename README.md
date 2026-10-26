# 🏠 House Price Prediction Using Linear Regression

A beginner-friendly machine learning project that predicts house prices using **Linear Regression** and provides a simple **Streamlit web application**.

## 📌 Project Overview

The model is trained using sample house data containing:

- Area in square feet
- Number of bedrooms
- Number of bathrooms
- Age of the house
- House price

The trained model is saved using Joblib and used by the Streamlit application to make predictions.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Linear Regression
- Joblib
- Streamlit

## 📁 Project Structure

```text
House-Price-Prediction/
│
├── app.py
├── train_model.py
├── house_price_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

Clone the repository and open the project folder.

Install the required libraries:

```bash
pip install -r requirements.txt
```

## 🤖 Train the Model

Run:

```bash
python train_model.py
```

This trains the Linear Regression model and creates:

```text
house_price_model.pkl
```

## 🌐 Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

The application allows the user to enter house details and displays the estimated price.

## 📊 Input Features

| Feature | Description |
|---|---|
| Area | House area in square feet |
| Bedrooms | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Age | Age of the house in years |

## 🎯 Output

The application predicts the estimated house price in **Indian Lakhs (₹ Lakhs)**.

## 🚀 Future Enhancements

- Use a larger real-world dataset
- Add more features such as location and parking
- Compare multiple machine learning algorithms
- Add graphs and data visualizations
- Deploy the Streamlit application online

## 👩‍💻 Project Type

Machine Learning / Data Science Mini Project
