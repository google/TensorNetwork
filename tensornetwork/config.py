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

default_backend = "tensorflow"

# for backwards compatibility default dtypes have to be `None`
# changing this will cause tests to fail due backend.convert_to_tensor(tensor)
# raising TypeErrors when incoming `tensor` has a dtype different from
# backend.dtype.

default_dtype = None

default_dtypes = {
    'tensorflow': None,
    'numpy': None,
    'pytorch': None,
    'jax': None,
    'shell': None
}
