import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
plt.style.use("seaborn-dark")

data = pd.read_csv('DataSets/score.csv')

plt.figure(figsize=(5, 5))
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.title("Student's Scores and Hours of Studying")
for i in range(0, data.shape[0]):
    plt.scatter(data['Hours'][i], data['Scores'][i], color='blue')

x = data['Hours'].values.reshape(-1, 1)
y = data['Scores'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=1/4, random_state=0)
lr = LinearRegression()
lr.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True,
                 n_jobs=1, normalize=False)
y_results = lr.predict(X_test)
plt.plot(X_test, y_results, color='red')
plt.show()
