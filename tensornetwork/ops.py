class _DefaultStack:

  def __init__(self):
    self.stack = []
    self._enforce_nesting = True

  def get_current_item(self):
    if not self.stack:
      return None
    return self.stack[-1]


_default_collection_stack = _DefaultStack()


def get_current_collection():
  return _default_collection_stack.get_current_item()
