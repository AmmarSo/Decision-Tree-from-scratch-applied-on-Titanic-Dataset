class Node:
    # Constructor
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, is_leaf=False, prediction=None, samples_count=None):
        """
        Represents a node in the decision tree.
        Can be either a decision node (with left/right children)
        or a leaf node (with a prediction).
        
        Parameters:
            feature_index (int): Index of the feature used to split (None if leaf)
            threshold (float or str): Threshold value for splitting
            left (Node): Left child node
            right (Node): Right child node
            is_leaf (bool): True if this node is a leaf
            prediction (any): Predicted class (if leaf)
            samples_count (int): Number of samples at this node
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.samples_count = samples_count

    # Method: print-friendly string representation
    def __str__(self):
        return f"[Leaf] Predict: {self.prediction}" if self.is_leaf else f"[Node] Feature {self.feature_index}, Threshold: {self.threshold}"

    # Optional: pure status (not currently used but can be helpful later)
    def is_pure(self):
        """
        Placeholder method for checking purity (optional).
        You can implement it if you store class counts in the node.
        """
        print("Next")

    # Optional: Describe method (useful for custom printing)
    def describe(self):
        """
        Prints a description of the node for debugging or visualization.
        """
        if self.is_leaf:
            print(f"Leaf node: Predict {self.prediction}, Samples: {self.samples_count}")
        else:
            print(f"Decision node: Feature {self.feature_index} <= {self.threshold}")
