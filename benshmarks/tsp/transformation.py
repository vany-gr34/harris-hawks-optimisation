import numpy as np 


class Transformation:
    def __init__(self, vector):
        self.vector = vector

    def forward_transform(self, Xi):
        pass
    def backward_transform(self, Xi):
        raise NotImplementedError("Subclasses must implement the inverse_transform method.")