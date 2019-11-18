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
    'edge',
    {
        mixins: [mixinGet, mixinGeometry],
        props: {
            edge: Array,
            state: Object
        },
        computed: {
            node1: function() {
                return this.getNode(this.edge[0][0]);
            },
            node2: function() {
                return this.getNode(this.edge[1][0]);
            },
            angle1: function() {
                return this.node1.axes[this.edge[0][1]].angle;
            },
            angle2: function() {
                return this.node2.axes[this.edge[1][1]].angle;
            },
            x1: function() {
                return this.node1.position.x + this.getAxisPoints(this.node1.axes[this.edge[0][1]].position, this.angle1, this.node1.rotation).x2;
            },
            y1: function() {
                return this.node1.position.y + this.getAxisPoints(this.node1.axes[this.edge[0][1]].position, this.angle1, this.node1.rotation).y2;
            },
            x2: function() {
                return this.node2.position.x + this.getAxisPoints(this.node2.axes[this.edge[1][1]].position, this.angle2, this.node2.rotation).x2;
            },
            y2: function() {
                return this.node2.position.y + this.getAxisPoints(this.node2.axes[this.edge[1][1]].position, this.angle2, this.node2.rotation).y2;
            }
        },
        template: `
            <g>
                <line class="edge" :x1="x1" :y1="y1" :x2="x2" :y2="y2"
                    stroke="#ddd" stroke-width="5" stroke-linecap="round" />
                <text v-if="edge[2]" :x="0.5 * (x1 + x2)" :y="0.5 * (y1 + y2)">
                    {{edge[2]}}
                </text>
            </g>
        `
    }
);

Vue.component(
    'proto-edge',
    {
        mixins: [mixinGeometry],
        props: {
            x: Number,
            y: Number,
            node: Object,
            axis: Number,
        },
        computed: {
            angle: function() {
                return this.node.axes[this.axis].angle;
            },
            x0: function() {
                return this.node.position.x + this.getAxisPoints(this.node.axes[this.axis].position, this.angle, this.node.rotation).x2;
            },
            y0: function() {
                return this.node.position.y + this.getAxisPoints(this.node.axes[this.axis].position, this.angle, this.node.rotation).y2;
            }
        },
        template: `
            <line class="proto-edge" :x1="x0" :y1="y0" :x2="x" :y2="y"
                stroke="#bbb" stroke-width="5" stroke-linecap="round" />
        `
    }
);
