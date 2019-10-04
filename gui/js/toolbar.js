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
        methods: {
            addAxis: function() {
                this.tensor.axes.push(null);
            },
            removeAxis: function() {
                if (this.tensor.axes.length < 1) {
                    return;
                }
                this.tensor.axes.pop();
                let oldAxis = this.tensor.axes.length;
                let tensor = this.tensor;
                let survingEdges = this.state.edges.filter(function(edge) {
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
                this.state.edges = survingEdges;
            }
        },
        computed: {
            tensor: function() {
                return this.state.selectedNode;
            }
        },
        template: `
            <div class="toolbar">
                <div v-if="tensor != null">
                    <h2>Tensor: {{tensor.name}}</h2>
                    <button @click="addAxis">Add Axis</button>
                    <button @click="removeAxis">Remove Axis</button>
                </div>
                <div v-else>Select a tensor to edit it</div>
            </div>
        `
    }
);
