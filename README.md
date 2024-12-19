# DVD Rental Duration Prediction

This project predicts the number of days a customer will rent a DVD based on various features. The prediction model helps a DVD rental company optimize inventory management by providing insights into rental durations.

## Problem Statement
The company provided data about DVD rentals and requested a regression model with a mean squared error (MSE) of 3 or less on the test set. This model aims to assist in inventory planning by forecasting rental durations effectively.

## Features in the Dataset
The dataset includes the following columns:
- `rental_date`: Date and time when the DVD was rented.
- `return_date`: Date and time when the DVD was returned.
- `amount`: Amount paid for renting the DVD.
- `amount_2`: Square of `amount`.
- `rental_rate`: Rental rate of the DVD.
- `rental_rate_2`: Square of `rental_rate`.
- `release_year`: Year of the movie's release.
- `length`: Length of the movie in minutes.
- `length_2`: Square of `length`.
- `replacement_cost`: Cost to replace the DVD.
- `special_features`: Special features like behind-the-scenes or deleted scenes.
- `NC-17`, `PG`, `PG-13`, `R`: Dummy variables for movie ratings (e.g., 1 for the corresponding rating, 0 otherwise).

## Models and Techniques
This project evaluates multiple regression models using pipelines:
- **Linear Regression**
- **Lasso Regression**
- **Gradient Boosting**
- **Random Forest**
- **AdaBoost**
- **K-Nearest Neighbors (KNN)**
- **XGBoost**

Each model's performance is measured using the mean squared error (MSE) on the test dataset. The best-performing model is selected for deployment.

## Implementation Steps
### 1. Data Preprocessing
- Convert `rental_date` and `return_date` to datetime objects.
- Compute rental duration (`rental_length_days`).
- Extract features from `special_features` (e.g., presence of behind-the-scenes content).
- Drop redundant columns like `special_features`, `rental_date`, and `return_date`.

### 2. Train-Test Split
The data is split into training and test sets using an 80-20 split:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)
```

### 3. Model Training
Pipelines are used to standardize features and fit models efficiently:
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipelines = {
    'linear_regression': Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())]),
    'lasso_regression': Pipeline([('scaler', StandardScaler()), ('regressor', Lasso(alpha=0.1))]),
    # Other models...
}
```

### 4. Model Evaluation
Each model's MSE is calculated, and the best model is identified:
```python
best_model = models[np.argmin(models_mse)]
```

## Results
The best model was **XGBoost**, achieving an MSE of **1.91**, exceeding the company’s expectations for accuracy.

## Prerequisites
- Python 3.x
- Libraries: pandas, NumPy, scikit-learn, XGBoost

To install dependencies:
```bash
pip install pandas numpy scikit-learn xgboost
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/your-username/dvd-rental-prediction.git
cd dvd-rental-prediction
```

2. Prepare the dataset:
   - Place `rental_info.csv` in the project directory.

3. Run the script:
```bash
python main.py
```

4. View the results in the console or export them to a file.

## Project Structure
```
.
├── rental_info.csv       # Dataset
├── preprocessing.py      # Data preprocessing scripts
├── model_training.py     # Model training scripts
├── evaluation.py         # Model evaluation scripts
├── main.py               # Entry point for running the project
├── README.md             # Project documentation
└── requirements.txt      # List of dependencies
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
- The DVD rental dataset provided by the Datacamp.
- Libraries and frameworks: scikit-learn, XGBoost, pandas, and NumPy.

Feel free to modify or extend the project as needed for your use case.
