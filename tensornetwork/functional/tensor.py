from collections import defaultdict

class Builder:
  def __init__(self, tensor, indices):
    self.tensor = tensor
    self.left = None
    self.right = None
    self.indices = indices

  def __init__(self, left, right):
    self.tensor = tensor
    self.left = left
    self.right = right
