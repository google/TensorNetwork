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

Vue.component(
    'toolbar',
    {
        props: {
            state: Object
        },
        data: function() {
            return {
                createTensorName: "",
            }
        },
        methods: {
            createTensor: function(event) {
                event.preventDefault();
                let workspace = document.getElementsByClassName('workspace')[0]
                    .getBoundingClientRect();

                let tensor = {
                    name: this.createTensorName,
                    position: {x: workspace.width / 2, y: workspace.height / 2},
                    axes: [null],
                    rotation: 0
                };

                this.state.tensors.push(tensor);
                this.state.selectedNode = tensor;

                this.createTensorName = "";
            },
            deleteTensor: function() {
                let selectedName = this.state.selectedNode.name;

                this.state.edges = this.state.edges.filter(function(edge) {
                    if (edge[0][0] === selectedName || edge[1][0] === selectedName) {
                        return false;
                    }
                    else {
                        return true;
                    }
                });
                this.state.tensors = this.state.tensors.filter(function(tensor) {
                    return tensor.name !== selectedName;
                });
                this.selectedNode = null;
            },
            addAxis: function() {
                this.tensor.rotation = 0;
                this.tensor.axes.push(null);
            },
            removeAxis: function() {
                this.tensor.rotation = 0;
                if (this.tensor.axes.length < 1) {
                    return;
                }
                this.tensor.axes.pop();
                let oldAxis = this.tensor.axes.length;
                let tensor = this.tensor;
                this.state.edges = this.state.edges.filter(function(edge) {
                    if (edge[0][0] === tensor.name && edge[0][1] === oldAxis) {
                        return false;
                    }
                    else if (edge[1][0] === tensor.name && edge[1][1] === oldAxis) {
                        return false;
                    }
                    else {
                        return true;
                    }
                });
            },
            rotateCounter: function() {
                this.tensor.rotation -= 2 * Math.PI / this.leastCommonMultiple(4, this.tensor.axes.length);
            },
            rotateClockwise: function() {
                this.tensor.rotation += 2 * Math.PI / this.leastCommonMultiple(4, this.tensor.axes.length);
            },
            leastCommonMultiple: function(a, b) {
                // assumes a, b are positive integers
                return a * b / this.greatestCommonDenominator(a, b);
            },
            greatestCommonDenominator: function(a, b) {
                while (b !== 0) {
                    let remainder = a % b;
                    a = b;
                    b = remainder;
                }
                return a;
            }
        },
        computed: {
            tensor: function() {
                return this.state.selectedNode;
            },
            nameTaken: function() {
                for (let i = 0; i < this.state.tensors.length; i++) {
                    if (this.createTensorName === this.state.tensors[i].name) {
                        return true;
                    }
                }
                return false;
            }
        },
        template: `
            <div class="toolbar">
                <h2>Create New Tensor</h2>
                <div class="button-holder">
                    <form @submit="createTensor">
                        <input type="text" v-model="createTensorName" />
                        <input type="submit" value="Create" :disabled="nameTaken" />
                    </form>
                </div>
                <div v-if="tensor != null">
                    <h2>Tensor: {{tensor.name}}</h2>
                    <div class="button-holder">
                        <button @click="deleteTensor">Delete</button>
                    </div>
                    <div class="button-holder">
                        <button @click="addAxis">Add Axis</button>
                        <button @click="removeAxis">Remove Axis</button>
                    </div>
                    <h4>Rotate</h4>
                    <div class="button-holder">
                        <button @click="rotateCounter">Counterclockwise</button>
                        <button @click="rotateClockwise">Clockwise</button>
                    </div>
                </div>
                <h2 v-else>Select a tensor to edit it</h2>
            </div>
        `
    }
);
