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
    renderLaTeX: true,
    selectedNodes: [],
    draggingNode: false,
    nodes: [
        {
            name: 't1',
            displayName: 't_1',
            size: [1, 1],
            axes: [
                {name: null, angle: 0, position: [0, 0]},
                {name: null, angle: Math.PI / 2, position: [0, 0]},
                {name: null, angle: Math.PI, position: [0, 0]},
            ],
			position: {x: 200, y: 300},
            rotation: 0,
            hue: 30
        },
        {
            name: 't2',
            displayName: 't_2',
            size: [1, 1],
            axes: [
                {name: null, angle: 0, position: [0, 0]},
                {name: null, angle: Math.PI / 2, position: [0, 0]},
                {name: null, angle: Math.PI, position: [0, 0]},
            ],
            position: {x: 367, y: 300},
            rotation: 0,
            hue: 30
        },
        {
            name: 't3',
            displayName: 't_3',
            size: [1, 1],
            axes: [
                {name: null, angle: 0, position: [0, 0]},
                {name: null, angle: Math.PI / 2, position: [0, 0]},
                {name: null, angle: Math.PI, position: [0, 0]},
            ],
            position: {x: 533, y: 300},
            rotation: 0,
            hue: 30
        },
        {
            name: 't4',
            displayName: 't_4',
            size: [1, 1],
            axes: [
                {name: null, angle: 0, position: [0, 0]},
                {name: null, angle: Math.PI / 2, position: [0, 0]},
                {name: null, angle: Math.PI, position: [0, 0]},
            ],
            position: {x: 700, y: 300},
            rotation: 0,
            hue: 30
        }
    ],
    edges: [
        [['t1', 0], ['t2', 2], null],
        [['t2', 0], ['t3', 2], null],
        [['t3', 0], ['t4', 2], null],
    ]
};
