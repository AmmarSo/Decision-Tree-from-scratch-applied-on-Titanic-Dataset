# main.py

import csv
from DecisionTree import DecisionTree
from Node import Node

# Step 1: Load and prepare the dataset
def load_titanic_data(file_path):
    X, y = [], []

    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                # Use selected features: Pclass, Sex, Age
                pclass = int(row['Pclass'])
                sex = 0 if row['Sex'] == 'male' else 1  # Encode: male = 0, female = 1
                age = float(row['Age']) if row['Age'] else None

                # Skip rows with missing age
                if age is None:
                    continue

                X.append([pclass, sex, age])
                y.append(int(row['Survived']))
            except:
                continue

    return X, y

# Step 2: Main execution
if __name__ == "__main__":
    # Load data
    X, y = load_titanic_data("data/Titanic-Dataset.csv")

    # Initialize and train the decision tree
    tree = DecisionTree(max_depth=3, min_samples_split=2, criterion="gini")
    tree.fit(X, y)

    # Step 3: Predict on training data (just to test)
    predictions = tree.predict(X)

    # Step 4: Compare predictions to actual results
    correct = sum(1 for i in range(len(y)) if y[i] == predictions[i])
    accuracy = correct / len(y)
    print(f"\nAccuracy on training data: {accuracy:.2f}")

    # Step 5: Print the tree
    print("\nDecision Tree:")
    tree.print_tree()
