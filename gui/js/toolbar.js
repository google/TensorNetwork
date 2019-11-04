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
                copyNodeName: '',
            }
        },
        methods: {
            deselectNode: function() {
                this.state.selectedNode = null;
            },
            deleteNode: function(event) {
                event.preventDefault();
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
            copyNode: function(event) {
                event.preventDefault();
                let workspace = document.getElementsByClassName('workspace')[0]
                    .getBoundingClientRect();

                let node = JSON.parse(JSON.stringify(this.node));
                node.name = this.copyNodeName;
                node.position = {x: workspace.width / 2, y: workspace.height / 2};

                this.state.nodes.push(node);
            },
            rotate: function(angle) {
                this.node.rotation += angle;
            }
        },
        computed: {
            node: function() {
                return this.state.selectedNode;
            },
            copyNodeDisabled: function() {
                return this.nameTaken || this.copyNodeName == null || this.copyNodeName === '';
            },
            nameTaken: function() {
                for (let i = 0; i < this.state.nodes.length; i++) {
                    if (this.copyNodeName === this.state.nodes[i].name) {
                        return true;
                    }
                }
                return false;
            }
        },
        template: `
            <div class="toolbar">
                <div v-if="node != null">
                    <section>
                        <div class="button-holder">
                            <button @click="deselectNode">Create new node</button>
                        </div>
                    </section>
                    <section>
                        <div>
                            <a class="delete" href="" @click="deleteNode(event)">delete</a>
                            <h2>Node: {{node.name}}</h2>
                        </div>
                        <div class="button-holder">
                            <h4>Copy Node</h4>
                            <form @submit="copyNode">
                                <input type="text" v-model="copyNodeName" placeholder="name of copy" />
                                <input type="submit" value="Copy" :disabled="copyNodeDisabled" />
                            </form>
                        </div>
                        <h4>Rotate</h4>
                            <button @click="rotate(-Math.PI / 4)">Counterclockwise</button>
                            <button @click="rotate(Math.PI / 4)">Clockwise</button>
                    </section>
                    <toolbar-edge-section :state="state" />
                    <toolbar-axis-section :state="state" />
                </div>
                <div v-else>
                    <tensor-creator :state="state" />
                    <section>
                        <h3>Select a node to edit it</h3>
                    </section>
                </div>
            </div>
        `
    }
);

Vue.component(
    'tensor-creator',
    {
        props: {
            state: Object
        },
        data: function() {
            return {
                nodeInitial: {
                    name: "",
                    axes: [],
                    position: {x: 100, y: 100},
                    rotation: 0,
                    hue: null
                },
                node: {},
                nodeShadow: {
                    name: "",
                    axes: [
                            {name: null, angle: 0},
                            {name: null, angle: Math.PI / 4},
                            {name: null, angle: Math.PI / 2},
                            {name: null, angle: 3 * Math.PI / 4},
                            {name: null, angle: Math.PI},
                            {name: null, angle: 5 * Math.PI / 4},
                            {name: null, angle: 3 * Math.PI / 2},
                            {name: null, angle: 7 * Math.PI / 4},
                    ],
                    position: {x: 100, y: 100},
                    rotation: 0,
                    hue: null
                },
                edges: [],
                width: 200,
                height: 200,
            };
        },
        created: function() {
            this.node = JSON.parse(JSON.stringify(this.nodeInitial));
            this.node.hue = Math.random() * 360;
        },
        methods: {
            createNode: function(event) {
                event.preventDefault();
                let workspace = document.getElementsByClassName('workspace')[0]
                    .getBoundingClientRect();

                this.node.position = {x: workspace.width / 2, y: workspace.height / 2};

                this.state.nodes.push(this.node);

                this.node = JSON.parse(JSON.stringify(this.nodeInitial));
                this.node.hue = Math.random() * 360;
            },
            onShadowAxisMouseDown: function(node, axis) {
                this.node.axes.push(JSON.parse(JSON.stringify(this.nodeShadow.axes[axis])));
            },
            onNodeAxisMouseDown: function(node, axis) {
                this.node.axes.splice(axis, 1);
            },
        },
        computed: {
            createNodeDisabled: function() {
                return this.nameTaken || this.node.name == null || this.node.name === '';
            },
            nameTaken: function() {
                for (let i = 0; i < this.state.nodes.length; i++) {
                    if (this.node.name === this.state.nodes[i].name) {
                        return true;
                    }
                }
                return false;
            }
        },
        template: `
            <section class="tensor-creator">
                <h2>Create New Node</h2>
                <div class="button-holder">
                    <form @submit="createNode">
                        <input type="text" v-model="node.name" placeholder="node name" />
                        <input type="submit" value="Create" :disabled="createNodeDisabled" />
                    </form>
                </div>
                <p>Click on an axis to add or remove it.</p>
                <div class="svg-container">
                    <svg class="workspace" xmlns="http://www.w3.org/2000/svg"
                        :width="width" :height="height">
                        <node :node="nodeShadow" :state="state" :disableDragging="true" :shadow="true"
                            @axismousedown="onShadowAxisMouseDown(node, ...arguments)" />
                        <node :node="node" :state="state" :disableDragging="true"
                            @axismousedown="onNodeAxisMouseDown(node, ...arguments)"/>
                    </svg>
                </div>
            </section>
            
        `
    }
)

Vue.component(
    'toolbar-edge-section',
    {
        props: {
            state: Object
        },
        methods: {
            deleteEdge: function(event, edge) {
                event.preventDefault();
                this.state.edges = this.state.edges.filter(function(candidate) {
                    return candidate !== edge;
                });
            }
        },
        computed: {
            node: function() {
                return this.state.selectedNode;
            }
        },
        template: `
            <section v-if="node != null">
                <h3>Edges</h3>
                <div v-for="edge in state.edges">
                    <div v-if="edge[0][0] === node.name || edge[1][0] === node.name">
                        <div>
                            <a class="delete" href="" @click="deleteEdge(event, edge)">delete</a>
                            <h4>{{edge[0][0]}}[{{edge[0][1]}}] to {{edge[1][0]}}[{{edge[1][1]}}]</h4>
                        </div>
                        <label for="edge-name-input">Name</label>
                        <input id="edge-name-input" type="text" v-model="edge[2]" placeholder="edge name" />
                    </div>
                </div>
            </section>
        `
    }
);

Vue.component(
    'toolbar-axis-section',
    {
        props: {
            state: Object
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
                        <h4>{{node.name}}[{{index}}]</h4>
                    </div>
                    <label for="axis-name-input">Name</label>
                    <input id="axis-name-input" type="text" v-model="node.axes[index].name" placeholder="axis name" />
                </div>
            </section>
        `
    }
);
