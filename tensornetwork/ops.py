# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class _DefaultNodeCollectionStack:
  """A stack to keep track of contexts that were entered."""

  def __init__(self):
    self.stack = []

  def get_current_item(self):
    return self.stack[-1] if self.stack else None


_default_collection_stack = _DefaultNodeCollectionStack()


def get_current_collection():
  return _default_collection_stack.get_current_item()
