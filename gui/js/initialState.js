// Copyright 2019 The TensorNetwork Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

let initialState = {
    draggingNode: false,
    tensors: [
        {
            name: 'A',
            axes: [null, 'a named axis'], // null values for axes correspond to unnamed axes
			position: {x: 100, y: 200},
            hue: 90
        },
        {
            name: 'B',
            axes: ['a named axis', 'foo', 'a free index'], // can have duplicate names for axes across tensors
			position: {x: 300, y: 300},
            hue: null
        }
    ],
    edges: [
        [['B', 1], ['A', 1], 'a named edge'], // optional third element describes edge name
    ]
};
