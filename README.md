# Algerian Forest Fire Predictor

An end-to-end Machine Learning project that predicts the **Fire Weather Index (FWI)** based on Algerian forest weather data. This solution handles multicollinearity using Ridge and Lasso regression and is deployed as a Flask web application.

## Project Overview
* **Goal:** Predict forest fire risks (FWI) using environmental features like Temperature, Humidity, Rain, and Wind Speed.
* **Accuracy:** Achieved **98.7% R² score** using Ridge Regression.
* **Techniques:** Solved high multicollinearity between features (DC, BUI, DMC) using Regularization (Lasso/Ridge).
* **Deployment:** Built a responsive web interface with **Flask** and deployed on **AWS EC2**.

## Tech Stack
* **Language:** Python 3.8+
* **Libraries:** Scikit-learn, Pandas, NumPy, Seaborn, Matplotlib
* **Web Framework:** Flask
* **Cloud:** AWS EC2 (Elastic Compute Cloud)

## Model Performance
I trained and compared three models to find the best fit:

| Model | R² Score | Description |
| :--- | :--- | :--- |
| **Linear Regression** | 98.9% | Baseline model (prone to overfitting due to multicollinearity). |
| **Lasso Regression** | 98.4% | Used L1 Regularization to feature selection. |
| **Ridge Regression** | **98.7%** | **(Selected Model)** Best balance of bias-variance trade-off. |

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Sohaib4238/Algerian_Forest_Project.git](https://github.com/Sohaib4238/Algerian_Forest_Project.git)
    cd Algerian_Forest_Project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Flask App:**
    ```bash
    python application.py
    ```

4.  **Access the App:**
    Open your browser and go to `http://127.0.0.1:5000/`

##Project Structure
```text
Algerian_Forest_Project/
├── dataset/                # Raw and Cleaned Data
├── models/                 # Pickled Models (ridge.pkl, scaler.pkl)
├── templates/              # HTML Frontend (index.html, home.html)
├── application.py          # Flask Main Application
├── training.ipynb          # Jupyter Notebook for EDA & Training
└── requirements.txt        # Python Dependencies
