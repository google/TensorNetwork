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
                    tensor: null,
                    axis: null,
                    dragging: false
                }
			};
		},
        methods: {
            onClick: function() {
                this.state.selectedNode = null;
            },
		    onAxisMouseDown: function(tensor, axis) {
                if (this.axisOccupied(tensor, axis)) {
                    return;
                }
                document.addEventListener('mousemove', this.dragAxis);
                document.addEventListener('mouseup', this.releaseAxisDrag);
                this.protoEdge.tensor = tensor;
                this.protoEdge.axis = axis;
            },
            dragAxis: function(event) {
                let workspace = document.getElementsByClassName('workspace')[0]
                    .getBoundingClientRect();
                this.protoEdge.dragging = true;
                this.protoEdge.x = event.pageX - workspace.left;
                this.protoEdge.y = event.pageY - workspace.top;
            },
            releaseAxisDrag: function() {
                document.removeEventListener('mousemove', this.dragAxis);
                document.removeEventListener('mouseup', this.releaseAxisDrag);
                this.protoEdge.dragging = false;
                this.protoEdge.tensor = null;
                this.protoEdge.axis = null;
            },
            onAxisMouseUp: function(tensor, axis) {
		        if (this.protoEdge.dragging) {
                    if (this.axisOccupied(tensor, axis)) {
                        return;
                    }
                    if (this.protoEdge.tensor.name === tensor.name
                        && this.protoEdge.axis === axis) {
                        return; // don't allow connection of an axis to itself
                    }
                    this.state.edges.push([
                        [this.protoEdge.tensor.name, this.protoEdge.axis],
                        [tensor.name, axis]
                    ])
                }
            },
            axisOccupied: function(tensor, axis) {
                for (let i = 0; i < this.state.edges.length; i++) {
                    let edge = this.state.edges[i];
                    if ((tensor.name === edge[0][0] && axis === edge[0][1])
                        || (tensor.name === edge[1][0] && axis === edge[1][1])) {
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
				    :tensor="protoEdge.tensor" :axis="protoEdge.axis" />
				<edge v-for="edge in state.edges" :edge="edge" :state="state" /> 
				<tensor v-for="tensor in state.tensors" :tensor="tensor" :state="state"
				    @axismousedown="onAxisMouseDown(tensor, ...arguments)"
				    @axismouseup="onAxisMouseUp(tensor, ...arguments)" />
			</svg>
		`
	}
);

let app = new Vue({
    el: '#app',
    data: {
        state: initialState // now state object is reactive, whereas initialState is not
    },
    template: `
        <div>
        <div class="app">
			<workspace :state="state" />
			<toolbar :state="state" />
        </div>
        <tensor-description :tensor="tensor" :state="state" v-for="tensor in state.tensors" />
        </div>

    `
});


