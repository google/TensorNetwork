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
    'workspace',
	{
        props: {
            state: Object
        },
        data: function() {
			return {
				width: 900,
				height: 600,
                protoEdge: {
                    x: null,
                    y: null,
                    node: null,
                    axis: null,
                    dragging: false
                }
			};
		},
        methods: {
            onClick: function() {
                this.state.selectedNodes = [];
            },
		    onAxisMouseDown: function(node, axis) {
                if (this.axisOccupied(node, axis)) {
                    return;
                }
                document.addEventListener('mousemove', this.dragAxis);
                document.addEventListener('mouseup', this.releaseAxisDrag);
                this.protoEdge.node = node;
                this.protoEdge.axis = axis;
            },
            dragAxis: function(event) {
                let workspace = document.getElementsByClassName('workspace')[0]
                    .getBoundingClientRect();
                this.protoEdge.dragging = true;
                this.protoEdge.x = event.clientX - workspace.left;
                this.protoEdge.y = event.clientY - workspace.top;
            },
            releaseAxisDrag: function() {
                document.removeEventListener('mousemove', this.dragAxis);
                document.removeEventListener('mouseup', this.releaseAxisDrag);
                this.protoEdge.dragging = false;
                this.protoEdge.node = null;
                this.protoEdge.axis = null;
            },
            onAxisMouseUp: function(node, axis) {
		        if (this.protoEdge.dragging) {
                    if (this.axisOccupied(node, axis)) {
                        return;
                    }
                    if (this.protoEdge.node.name === node.name
                        && this.protoEdge.axis === axis) {
                        return; // don't allow connection of an axis to itself
                    }
                    this.state.edges.push([
                        [this.protoEdge.node.name, this.protoEdge.axis],
                        [node.name, axis],
                        null
                    ])
                }
            },
            axisOccupied: function(node, axis) {
                for (let i = 0; i < this.state.edges.length; i++) {
                    let edge = this.state.edges[i];
                    if ((node.name === edge[0][0] && axis === edge[0][1])
                        || (node.name === edge[1][0] && axis === edge[1][1])) {
                        return true;
                    }
                }
                return false;
            }
        },
		template: `
			<svg class="workspace" xmlns="http://www.w3.org/2000/svg"
			    :width="width" :height="height" @click="onClick">
                <proto-edge v-if="protoEdge.dragging" :x="protoEdge.x" :y="protoEdge.y"
				    :node="protoEdge.node" :axis="protoEdge.axis" />
				<edge v-for="edge in state.edges" :edge="edge" :state="state" /> 
				<node v-for="node in state.nodes" :node="node" :state="state"
				    @axismousedown="onAxisMouseDown(node, ...arguments)"
				    @axismouseup="onAxisMouseUp(node, ...arguments)" />
			</svg>
		`
	}
);

