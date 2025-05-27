from Node import Node

class DecisionTree:

    def __init__(self, root=None, max_depth=None,  min_samples_split=None, criterion=None, n_classes=None):
        self.root = root
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split	
        self.criterion = criterion
        self.n_classes = n_classes

    def fit(self, X, y):
        # Store number of classes if not already set
        if self.n_classes is None:
            self.n_classes = len(set(y))
        
        # Build the tree and set the root node
        self.root = self.build_tree(X, y, depth=0)

    def build_tree(self, X, y, depth):
        # Count the number of samples in this node
        num_samples = len(y)

        # Count the number of unique classes in this node
        unique_classes = list(set(y))

        # 1. If all samples belong to the same class → pure node → return a leaf
        if len(unique_classes) == 1:
            return Node(is_leaf=True, prediction=unique_classes[0], samples_count=num_samples)

        # 2. If stopping criteria is met → return a leaf node with majority class
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (self.min_samples_split is not None and num_samples < self.min_samples_split):
            majority_class = max(set(y), key=y.count)
            return Node(is_leaf=True, prediction=majority_class, samples_count=num_samples)

        # 3. Try to find the best possible split
        split = self.best_split(X, y)

        if split is None:
            # No valid split found → return leaf
            majority_class = max(set(y), key=y.count)
            return Node(is_leaf=True, prediction=majority_class, samples_count=num_samples)

        # Unpack the split result
        feature_index, threshold, X_left, y_left, X_right, y_right = split

        # 4. Recursively build left and right branches
        left_node = self.build_tree(X_left, y_left, depth + 1)
        right_node = self.build_tree(X_right, y_right, depth + 1)

        # 5. Return a decision node
        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_node,
            right=right_node,
            is_leaf=False,
            samples_count=num_samples
        )

    def best_split(self, X, y):
        # Initialize best gain and best split container
        best_gain = 0
        best_split = None

        # Calculate impurity of current node
        impurity_parent = self.gini(y) if self.criterion == "gini" else self.entropy(y)

        # Loop over every feature
        for feature_index in range(len(X[0])):
            # Get all unique values (potential thresholds)
            thresholds = sorted(set(row[feature_index] for row in X))

            for threshold in thresholds:
                # Split dataset into left and right groups
                X_left, y_left, X_right, y_right = self.split_dataset(X, y, feature_index, threshold)

                # Skip if one of the sides is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate impurity of children
                impurity_left = self.gini(y_left) if self.criterion == "gini" else self.entropy(y_left)
                impurity_right = self.gini(y_right) if self.criterion == "gini" else self.entropy(y_right)

                weight_left = len(y_left) / len(y)
                weight_right = len(y_right) / len(y)

                impurity_children = weight_left * impurity_left + weight_right * impurity_right

                # Calculate gain
                gain = impurity_parent - impurity_children

                # Update best split
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, X_left, y_left, X_right, y_right)

        return best_split

    def split_dataset(self, X, y, feature_index, threshold):
        X_left, y_left, X_right, y_right = [], [], [], []

        for xi, yi in zip(X, y):
            if xi[feature_index] <= threshold:
                X_left.append(xi)
                y_left.append(yi)
            else:
                X_right.append(xi)
                y_right.append(yi)

        return X_left, y_left, X_right, y_right

    def predict(self, X):
        return [self.predict_sample(x, self.root) for x in X]
    
    def predict_sample(self, x, node):
        # If we reach a leaf → return prediction
        if node.is_leaf:
            return node.prediction

        # Decide to go left or right
        if x[node.feature_index] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def gini(self, y):
        from collections import Counter
        counts = Counter(y)
        impurity = 1
        for label in counts:
            prob = counts[label] / len(y)
            impurity -= prob ** 2
        return impurity

    def entropy(self, y):
        from collections import Counter
        import math
        counts = Counter(y)
        entropy = 0
        for label in counts:
            prob = counts[label] / len(y)
            entropy -= prob * math.log2(prob)
        return entropy

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.root

        indent = "  " * depth
        if node.is_leaf:
            print(f"{indent}Predict: {node.prediction} ({node.samples_count} samples)")
        else:
            print(f"{indent}Feature {node.feature_index} <= {node.threshold}?")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)
