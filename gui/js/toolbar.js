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
                this.state.selectedNodes = [];
            },
            deleteNode: function(event) {
                event.preventDefault();
                let selectedName = this.state.selectedNodes[0].name;

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
                this.selectedNodes = [];
            },
            copyNode: function(event) {
                event.preventDefault();
                let workspace = document.getElementsByClassName('workspace')[0]
                    .getBoundingClientRect();

                let node = JSON.parse(JSON.stringify(this.node));
                node.name = this.copyNodeName;
                node.position = {x: workspace.width / 2, y: workspace.height / 2};

                this.state.nodes.push(node);
                this.state.selectedNodes = [node];
                this.copyNodeName = '';
            },
            rotate: function(angle) {
                this.node.rotation += angle;
            }
        },
        computed: {
            node: function() {
                return this.state.selectedNodes[0];
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
                <div v-if="state.selectedNodes.length === 0">
                    <tensor-creator :state="state" />
                    <section>
                        <h3>Selecting nodes</h3>
                        <p>Click a node to select it for editing.</p>
                        <p>Drag-select or shift-click multiple nodes to drag as a group and adjust alignment and
                        spacing.</p>
                    </section>
                </div>
                <div v-else-if="state.selectedNodes.length === 1">
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
                        <h4>Set LaTeX Label</h4>
                        <input type="text" v-model="node.displayName" placeholder="LaTeX label" />
                        <h4>Copy Node</h4>
                        <form @submit="copyNode">
                            <input type="text" v-model="copyNodeName" placeholder="name of copy" />
                            <input type="submit" value="Copy" :disabled="copyNodeDisabled" />
                        </form>
                        <h4>Rotate</h4>
                        <button @click="rotate(-Math.PI / 4)">Counterclockwise</button>
                        <button @click="rotate(Math.PI / 4)">Clockwise</button>
                    </section>
                    <toolbar-edge-section :state="state" />
                    <toolbar-axis-section :state="state" />
                </div>
                <div v-else>
                    <toolbar-multinode-section :state="state" />
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
                size1: 1,
                size2: 1,
                hue: null,
                node: {},
                width: 250,
                height: 250,
            };
        },
        created: function() {
            this.reset();
        },
        watch: {
            size1: function() {
                this.reset();
            },
            size2: function() {
                this.reset();
            },
        },
        methods: {
            reset: function() {
                this.hue = Math.random() * 360;
                this.node = JSON.parse(JSON.stringify(this.nodeInitial));
            },
            createNode: function(event) {
                event.preventDefault();
                let workspace = document.getElementsByClassName('workspace')[0]
                    .getBoundingClientRect();

                this.node.position = {x: workspace.width / 2, y: workspace.height / 2};

                this.state.nodes.push(this.node);
                this.reset();
            },
            onShadowAxisMouseDown: function(node, axis) {
                this.node.axes.push(JSON.parse(JSON.stringify(this.nodeShadow.axes[axis])));
            },
            onNodeAxisMouseDown: function(node, axis) {
                this.node.axes.splice(axis, 1);
            },
            axes: function(size1, size2) {
                let makeAxis = function(direction, position) {
                    return {name: null, angle: direction * Math.PI / 4, position: position};
                };
                let x_end = Math.min((size1 - 1) / 2, 1);
                let y_end = Math.min((size2 - 1) / 2, 1);
                let output = [];
                for (let n = 0; n < size1; n++) {
                    let x = size1 !== 1 ? (-x_end * (size1 - 1 - n) + x_end * n) / (size1 - 1) : 0; // Avoid div by 0
                    for (let m = 0; m < size2; m++) {
                        let y = size2 !== 1 ? (-y_end * (size2 - 1 - m) + y_end * m) / (size2 - 1) : 0;
                        if (n === 0) {
                            output.push(makeAxis(4, [x, y]))
                        }
                        if (n === size1 - 1) {
                            output.push(makeAxis(0, [x, y]))
                        }
                        if (m === 0) {
                            output.push(makeAxis(6, [x, y]))
                        }
                        if (m === size2 - 1) {
                            output.push(makeAxis(2, [x, y]))
                        }
                        if (n === 0 && m === 0) {
                            output.push(makeAxis(5, [x, y]))
                        }
                        if (n === 0 && m === size2 - 1) {
                            output.push(makeAxis(3, [x, y]))
                        }
                        if (n === size1 - 1 && m === 0) {
                            output.push(makeAxis(7, [x, y]))
                        }
                        if (n === size1 - 1 && m === size2 - 1) {
                            output.push(makeAxis(1, [x, y]))
                        }

                    }
                }
                return output;
            }
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
            },
            nodeInitial: function() {
                return {
                    name: "",
                    size: [parseFloat(this.size1), parseFloat(this.size2)],
                    axes: [],
                    position: {x: 125, y: 125},
                    rotation: 0,
                    hue: this.hue
                };
            },
            nodeShadow: function() {
                return {
                    name: "",
                    size: [parseFloat(this.size1), parseFloat(this.size2)],
                    axes: this.axes(parseFloat(this.size1), parseFloat(this.size2)),
                    position: {x: 125, y: 125},
                    rotation: 0,
                    hue: null
                };
            },
            renderLaTeX: function() {
                return this.state.renderLaTeX && window.MathJax;

            }
        },
        template: `
            <section class="tensor-creator">
                <h2>Create New Node</h2>
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
                    <label>Width {{size1}}</label>
                    <input type="range" v-model="size1" min="1" max="7" step="1" class="slider" id="myRange">
                </div>
                </div>
                    <label>Height {{size2}}</label>
                    <input type="range" v-model="size2" min="1" max="7" step="1" class="slider" id="myRange">
                </div>
                <div class="button-holder">
                    <form @submit="createNode">
                        <input type="text" v-model="node.name" placeholder="node name" />
                        <input type="text" v-if="renderLaTeX" v-model="node.displayName" placeholder="LaTeX label" />
                        <input type="submit" value="Create" :disabled="createNodeDisabled" />
                    </form>
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
                return this.state.selectedNodes[0];
            }
        },
        template: `
            <section>
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
                return this.state.selectedNodes[0];
            }
        },
        template: `
            <section>
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

Vue.component(
    'toolbar-multinode-section',
    {
        props: {
            state: Object
        },
        data: function() {
            return {
                alignmentY: null,
                alignmentX: null,
                spacingY: null,
                spacingX: null
            }
        },
        created: function() {
            this.alignmentY = this.state.selectedNodes[0].position.y;
            this.alignmentX = this.state.selectedNodes[0].position.x;
            this.spacingY = this.state.selectedNodes[1].position.y - this.state.selectedNodes[0].position.y;
            this.spacingX = this.state.selectedNodes[1].position.x - this.state.selectedNodes[0].position.x;
        },
        methods: {
            alignVertically: function(event) {
                event.preventDefault();
                for (let i = 0; i < this.state.selectedNodes.length; i++) {
                    this.state.selectedNodes[i].position.y = parseFloat(this.alignmentY);
                }
            },
            alignHorizontally: function(event) {
                event.preventDefault();
                for (let i = 0; i < this.state.selectedNodes.length; i++) {
                    this.state.selectedNodes[i].position.x = parseFloat(this.alignmentX);
                }
            },
            spaceVertically: function(event) {
                event.preventDefault();
                let baseline = this.state.selectedNodes[0].position.y;
                for (let i = 1; i < this.state.selectedNodes.length; i++) {
                    this.state.selectedNodes[i].position.y = baseline + i * parseFloat(this.spacingY);
                }
            },
            spaceHorizontally: function(event) {
                event.preventDefault();
                let baseline = this.state.selectedNodes[0].position.x;
                for (let i = 1; i < this.state.selectedNodes.length; i++) {
                    this.state.selectedNodes[i].position.x = baseline + i * parseFloat(this.spacingX);
                }
            },
            disabledFor: function(length) {
                return length == null || length == "" || isNaN(parseFloat(length));
            }
        },
        template: `
            <div>
                <section>
                    <h2>Multiple Nodes</h2>
                    <div v-for="node in state.selectedNodes">
                        <p><strong>{{node.name}}</strong> - <em>x</em>: {{node.position.x}}, <em>y</em>: {{node.position.y}}</p>
                    </div>
                    <em>Shift-click a node in the workspace to deselect it.</em>
                </section>
                <section>
                    <h3>Align Vertically</h3>
                    <form @submit="alignVertically">
                        <input type="number" v-model="alignmentY" />
                        <input type="submit" value="Align" :disabled="disabledFor(alignmentY)" />
                    </form>
                </section>
                <section>
                    <h3>Align Horizontally</h3>
                    <form @submit="alignHorizontally">
                        <input type="number" v-model="alignmentX" />
                        <input type="submit" value="Align" :disabled="disabledFor(alignmentX)" />
                    </form>
                </section>
                <section>
                    <h3>Space Vertically</h3>
                    <form @submit="spaceVertically">
                        <input type="number" v-model="spacingY" />
                        <input type="submit" value="Space" :disabled="disabledFor(spacingY)" />
                    </form>
                </section>
                <section>
                    <h3>Space Horizontally</h3>
                    <form @submit="spaceHorizontally">
                        <input type="number" v-model="spacingX" />
                        <input type="submit" value="Space" :disabled="disabledFor(spacingX)" />
                    </form>
                </section>
            </div>
        `
    }
);
