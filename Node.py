class Node:
    # Attributes
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, is_leaf=False, prediction=None, samples_count=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.samples_count = samples_count
    
    # Method
    def __str__(self):
        return f"[Leaf] Predict: {self.prediction}" if self.is_leaf else f"[Node] Feature {self.feature_index}, Threshold: {self.threshold}"
    

    def is_pure(self):
        print("Next")
    
    def describe(self):
        print("Next")
