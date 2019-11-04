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
    selectedNode: null,
    draggingNode: false,
    nodes: [
        {
            name: 'A',
            axes: [null, null],
			position: {x: 100, y: 200},
            rotation: 0,
            hue: null
        },
        {
            name: 'B',
            axes: [null, null, null],
			position: {x: 300, y: 300},
            rotation: Math.PI / 2,
            hue: null
        }
    ],
    edges: [
        [['B', 1], ['A', 1], null],
    ]
};
