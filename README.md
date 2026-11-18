# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries such as Pandas and Scikit-learn.
2. Load the Salary.csv dataset and select Level and Salary.
3. Create and train the Decision Tree Regressor model.
4. Predict salaries and evaluate the model using MSE and R² score.
5. Display actual and predicted salaries and predict the salary for a new employee. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Udhaya Nandhini M
RegisterNumber:  212225240177
*/
```
```
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Salary.csv")

print("Employee Salary Dataset:")
print(data)

X = data[["Level"]]

y = data["Salary"]

model = DecisionTreeRegressor(
    random_state=42
)

model.fit(X, y)

y_pred = model.predict(X)

print("\nActual Salaries:")
print(y.values)

print("\nPredicted Salaries:")
print(y_pred)

mse = mean_squared_error(y, y_pred)


r2 = r2_score(y, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)


new_level = pd.DataFrame({
    "Level": [6]
})

predicted_salary = model.predict(new_level)

print("\nNew Employee Level:", 6)
print("Predicted Salary:", predicted_salary[0])
```
## Output:
![Decision Tree Regressor Model for Predicting the Salary of the Employee](sam.png)
![alt text](<Screenshot 2026-11-18 085512.png>)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
