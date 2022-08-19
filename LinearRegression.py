import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
plt.style.use("seaborn-dark")

data = pd.read_csv('DataSets/score.csv')


def CreateVisuals(dataFrame):
    plt.figure(figsize=(10, 10))
    plt.xlabel("Hours")
    plt.ylabel("Scores")
    plt.title("Student's Scores and Hours of Studying")
    for i in range(0, data.shape[0]):
        plt.scatter(data['Hours'][i], data['Scores'][i], color='Green')
    plt.plot()
    return plt


def Init(dataFrame):
    plot = CreateVisuals(data)
    plot.show()

# def TrainandTest(DataFrame):
#     X, y = data['Hours'], data['Scores']
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=1/4, random_state=0)
