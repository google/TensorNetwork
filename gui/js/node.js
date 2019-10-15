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

let mixinNode = {
	props: {
        node: Object,
    	state: Object,
    },
    methods: {
		neighborAt: function(axis) {
			for (let i = 0; i < this.neighbors.length; i++) {
				if (this.neighbors[i].axis === axis) {
					return this.neighbors[i].neighbor;
				}
			}
			return null;
		},
		edgeNameAt: function(axis) {
			for (let i = 0; i < this.neighbors.length; i++) {
				if (this.neighbors[i].axis === axis) {
					return this.neighbors[i].edgeName;
				}
			}
			return null;
		}
    },
	computed: {
        neighbors: function() {
			return this.getNeighborsOf(this.node.name);
		}
	},
};

Vue.component(
    'node',
	{
		mixins: [mixinGet, mixinGeometry, mixinNode],
        data: function() {
		    return {
		        mouse: {
                    x: null,
                    y: null
                }
            }
        },
        methods: {
		    onClick: function(event) {
		        event.stopPropagation();
            },
		    onMouseDown: function(event) {
                this.state.selectedNode = this.node;

		        document.addEventListener('mousemove', this.onMouseMove);
		        document.addEventListener('mouseup', this.onMouseUp);
                this.state.draggingNode = true;

		        this.mouse.x = event.pageX;
		        this.mouse.y = event.pageY;
            },
            onMouseMove: function(event) {
                let dx = event.pageX - this.mouse.x;
                let dy = event.pageY - this.mouse.y;
                this.node.position.x += dx;
                this.node.position.y += dy;
                this.mouse.x = event.pageX;
                this.mouse.y = event.pageY;
            },
            onMouseUp: function() {
                document.removeEventListener('mousemove', this.onMouseMove);
                document.removeEventListener('mouseup', this.onMouseUp);

                this.state.draggingNode = false;

                let workspace = document.getElementsByClassName('workspace')[0]
                    .getBoundingClientRect();
                if (this.node.position.x < this.nodeWidth / 2) {
                    this.node.position.x = this.nodeWidth / 2;
                }
                if (this.node.position.y < this.nodeHeight / 2) {
                    this.node.position.y = this.nodeHeight / 2;
                }
                if (this.node.position.x > workspace.width - this.nodeWidth / 2) {
                    this.node.position.x = workspace.width - this.nodeWidth / 2;
                }
                if (this.node.position.y > workspace.height - this.nodeHeight / 2) {
                    this.node.position.y = workspace.height - this.nodeHeight / 2;
                }
            },
            onAxisMouseDown: function(axis) {
		        this.$emit('axismousedown', axis);
            },
            onAxisMouseUp: function(axis) {
		        this.$emit('axismouseup', axis);
            }
        },
		computed: {
			translation: function() {
				return 'translate(' + this.node.position.x + ' ' + this.node.position.y + ')';
			},
            brightness: function() {
			    if (this.state.selectedNode != null && this.state.selectedNode.name === this.node.name) {
                    return 50;
                }
			    else {
			        return 80;
                }
            },
			style: function() {
				return 'fill: hsl(' + this.node.hue + ', 80%, ' + this.brightness + '%);'
			}
		},
        created: function() {
		    if (this.node.hue == null) {
		        this.node.hue = Math.random() * 360;
            }
        },
		template: `
			<g class="node" :transform="translation" 
                @click="onClick" @mousedown="onMouseDown" @mouseup="onMouseUp">
			    <axis v-for="(axisName, i) in node.axes" :node="node" :index="i"
			        :state="state" @axismousedown="onAxisMouseDown(i)"
			        @axismouseup="onAxisMouseUp(i)"/>
				<rect :x="-nodeWidth / 2" :y="-nodeHeight / 2" :width="nodeWidth"
				    :height="nodeHeight" :rx="nodeCornerRadius" :style="style" />
				<text x="0" y="0">{{node.name}}</text>
			</g>
		`	
	}
);

Vue.component(
    'axis',
    {
        mixins: [mixinGet, mixinGeometry, mixinNode],
        props: {
            node: Object,
            index: Number,
            state: Object,
        },
        data: function() {
            return {
                dragging: false,
                brightness: 80
            }
        },
        methods: {
            onMouseDown: function(event) {
                event.stopPropagation();
                this.$emit('axismousedown');
                this.dragging = true;
                document.addEventListener('mouseup', this.onDragEnd);
            },
            onMouseUp: function() {
                this.$emit('axismouseup');
            },
            onDragEnd: function() {
                this.dragging = false;
                document.removeEventListener('mouseup', this.onDragEnd);
            },
            onMouseEnter: function() {
                if (this.state.draggingNode) {
                    return;
                }
                if (this.neighborAt(this.index) != null) {
                    return; // don't highlight an axis that is occupied
                }
                if (this.dragging) {
                    return; // don't highlight self if self is being dragged
                }
                this.brightness = 50;
            },
            onMouseLeave: function() {
                this.brightness = 80;
            },
        },
        computed: {
            nAxes: function() {
                return this.node.axes.length;
            },
            angle: function() {
                return this.axisAngle(this.index, this.nAxes) + this.node.rotation;
            },
            x: function() {
                return this.axisX(this.angle);
            },
            y: function() {
                return this.axisY(this.angle);
            },
            stroke: function() {
                return 'hsl(' + this.node.hue + ', 80%, ' + this.brightness + '%)';
            }
        },
        template: `
            <line class="axis" x1="0" y1="0" :x2="x" :y2="y" :stroke="stroke"
                stroke-width="5" stroke-linecap="round"
                @mousedown="onMouseDown" @mouseup="onMouseUp"
                @mouseenter="onMouseEnter" @mouseleave="onMouseLeave"/>
        `
    }
);

Vue.component(
    'node-description',
    {
        mixins: [mixinGet, mixinNode],
        template: `
            <p>Node {{node.name}} has {{node.axes.length}} axes:
                <ul>
                    <li v-for="(axisName, i) in node.axes">
                        Axis {{i}} <span v-if="axisName">({{axisName}})</span>
                        <span v-if="neighborAt(i)">is connected to axis {{neighborAt(i)[1]}}
                            <span v-if="getAxis(neighborAt(i))">({{getAxis(neighborAt(i))}})</span>
                            of node {{getNode(neighborAt(i)[0]).name}}
                            <span v-if="edgeNameAt(i)">by edge "{{edgeNameAt(i)}}"</span>
                        </span>
                        <span v-else>is free</span>
                    </li>
                </ul>
            </p>
        `
    }
);
