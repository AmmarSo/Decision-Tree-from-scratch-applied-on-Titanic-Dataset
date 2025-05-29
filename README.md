# 🌳 Titanic Survivors Classification using Decision Tree (from Scratch)
This project demonstrates how to implement a Decision Tree classifier from scratch using the famous Titanic dataset.
It is designed for beginners who want to understand the core logic behind one of the most interpretable machine learning algorithms — without using any ML libraries like scikit-learn.

## 🧾 Full Explanation on Medium: coming soon

## 📊 Dataset
- Name: Titanic - Machine Learning from Disaster

- Source: Kaggle Titanic Competition

- Samples: 891 passengers (training set)

- Features (after preprocessing):

- Pclass (Passenger Class)

- Sex (converted to binary: 0 = male, 1 = female)

- Age (numerical, with missing values filled using the median)

- Survived (0 = No, 1 = Yes)

## 📁 Files
- DecisionTree.py – contains the full tree-building implementation

- Node.py – defines the tree’s structure (leaf and decision nodes)

- main.py – runs the training, evaluation, and displays the decision tree

- titanic.csv – the dataset used for training

- README.md – this file

##🧠 Why from Scratch?
Building machine learning models from scratch helps you understand what’s happening under the hood:

How a tree selects the best splits

How Gini impurity works

How recursive tree construction operates

How predictions are made by traversing the tree

Once you understand that, using tools like scikit-learn becomes much more meaningful.

✅ Requirements
Python 3.x

pandas

(Optional) matplotlib or graphviz for visualization
