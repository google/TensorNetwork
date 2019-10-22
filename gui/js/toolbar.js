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
                createNodeName: "",
            }
        },
        methods: {
            createNode: function(event) {
                event.preventDefault();
                let workspace = document.getElementsByClassName('workspace')[0]
                    .getBoundingClientRect();

                let node = {
                    name: this.createNodeName,
                    position: {x: workspace.width / 2, y: workspace.height / 2},
                    axes: [null],
                    rotation: 0
                };

                this.state.nodes.push(node);
                this.state.selectedNode = node;

                this.createNodeName = "";
            },
            deleteNode: function() {
                let selectedName = this.state.selectedNode.name;

                this.state.edges = this.state.edges.filter(function(edge) {
                    if (edge[0][0] === selectedName || edge[1][0] === selectedName) {
                        return false;
                    }
                    else {
                        return true;
                    }
                });
                this.state.nodes = this.state.nodes.filter(function(node) {
                    return node.name !== selectedName;
                });
                this.selectedNode = null;
            },
            rotateCounter: function() {
                this.node.rotation -= 2 * Math.PI / this.leastCommonMultiple(4, this.node.axes.length);
            },
            rotateClockwise: function() {
                this.node.rotation += 2 * Math.PI / this.leastCommonMultiple(4, this.node.axes.length);
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
            node: function() {
                return this.state.selectedNode;
            },
            createNodeDisabled: function() {
                return this.nameTaken || this.createNodeName == null || this.createNodeName === '';
            },
            nameTaken: function() {
                for (let i = 0; i < this.state.nodes.length; i++) {
                    if (this.createNodeName === this.state.nodes[i].name) {
                        return true;
                    }
                }
                return false;
            }
        },
        template: `
            <div class="toolbar">
                <section>
                    <h2>Create New Node</h2>
                    <div class="button-holder">
                        <form @submit="createNode">
                            <input type="text" v-model="createNodeName" placeholder="node name" />
                            <input type="submit" value="Create" :disabled="createNodeDisabled" />
                        </form>
                    </div>
                </section>
                <div v-if="node != null">
                    <section>
                        <div>
                            <a class="delete" href="" @click="deleteNode">delete</a>
                            <h2>Node: {{node.name}}</h2>
                        </div>
                        <h4>Rotate</h4>
                            <div class="button-holder">
                                <button @click="rotateCounter">Counterclockwise</button>
                                <button @click="rotateClockwise">Clockwise</button>
                            </div>
                    </section>
                    <toolbar-axis-section :state="state" />
                </div>
                <section v-else>
                    <h2>Select a node to edit it</h2>
                </section>
            </div>
        `
    }
);

Vue.component(
    'toolbar-axis-section',
    {
        props: {
            state: Object
        },
        methods: {
            addAxis: function() {
                this.node.axes.push(null);
            },
            deleteAxis: function(event, index) {
                event.preventDefault();
                this.node.axes.splice(index, 1);
            }
        },
        computed: {
            node: function() {
                return this.state.selectedNode;
            }
        },
        template: `
            <section v-if="node != null">
                <h3>Axes</h3>
                <div v-for="(axis, index) in node.axes">
                    <div>
                        <a class="delete" href="" @click="deleteAxis(event, index)">delete</a>
                        <h4>{{node.name}}[{{index}}]</h4>
                    </div>
                    <label for="axis-name-input">Name</label>
                    <input id="axis-name-input" type="text" v-model="node.axes[index]" placeholder="axis name" />
                </div>
                <div class="button-holder">
                    <button @click="addAxis">Add axis</button>
                </div>
            </section>
        `
    }
);
