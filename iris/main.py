#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from data_process import process_iris_data_to_csv


def process_classes(iris_data):
    unique_classes = iris_data["class"].unique().tolist()
    return iris_data.replace(unique_classes, list(range(len(unique_classes))))


if __name__ == "__main__":
    f_iris_data = "./dataset/iris.data"
    f_iris_csv = "./dataset/iris.csv"
    columns = ["sepal_l", "sepal_w", "petal_l", "petal_w", "class"]
    K_FOLD = 5

    process_iris_data_to_csv(f_iris_data, f_iris_csv, columns)

    iris_data = pd.read_csv(f_iris_csv)
    print(iris_data.head())
    print(iris_data.describe())
    # visualize the whole dataset, find some relations
    sns.pairplot(iris_data, hue='class')
    iris_data.hist(edgecolor = 'black', linewidth=1.2, figsize=(15,5))

    # prepare data
    iris_data = process_classes(iris_data)
    X = iris_data.drop(["class"], axis=1)
    y = iris_data["class"]

    # train : test = 0.8 : 0.2
    print("\n--- Begin ---")
    accuracy_scores = dict()

    clfs = ["KNN", "LR", "SVC", "DecisionTree"]
    models = [
        KNeighborsClassifier(n_neighbors=len(iris_data["class"].unique().tolist())),
        LogisticRegression(),
        SVC(),
        DecisionTreeClassifier()
    ]

    for clf, model in zip(clfs, models):
        print(f"--- Model: {clf} ---")
        scores = list()
        for fold in range(K_FOLD):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(metrics.accuracy_score(y_test, y_pred))

            if fold == 0:
                y_pred = model.predict(X_test)
                print(classification_report(y_test, y_pred))
                print(confusion_matrix(y_test, y_pred))
        accuracy_scores[clf] = sum(scores) / K_FOLD
        print(f"{clf}'s accuracy is {accuracy_scores[clf]}.")
        print()

    print("\n--- End ---")

    # %%
