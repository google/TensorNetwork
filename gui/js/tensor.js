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

let mixinTensor = {
	props: {
        tensor: Object,
    	state: Object,
    },
    methods: {
        getNeighborsOf: function(name) {
            let neighbors = [];
            let edges = this.state.edges;
            for (let i = 0; i < edges.length; i++) {
                let edge = edges[i];
                if (edge[0][0] === name) {
                    neighbors.push({
                        axis: edge[0][1],
                        neighbor: edge[1],
                        edgeName: edge[2]
                    });
                }
                else if (edge[1][0] === name) {
                    neighbors.push({
                        axis: edge[1][1],
                        neighbor: edge[0],
                        edgeName: edge[2]
                    });
                }
            }
            return neighbors;
        },
        getTensor: function(name) {
            for (let i = 0; i < this.state.tensors.length; i++) {
                if (this.state.tensors[i].name === name) {
                    return this.state.tensors[i];
                }
            }
            return null;
        },
        getAxis: function(address) {
            let [tensorName, axisIndex] = address;
            let tensor = this.getTensor(tensorName);
            return tensor.axes[axisIndex];
        },
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
			return this.getNeighborsOf(this.tensor.name);
		}
	},
};

Vue.component(
    'tensor',
	{
		mixins: [mixinTensor],
        data: function() {
		    return {
		        mouse: {
                    x: null,
                    y: null
                },
                width: 50,
                height: 50,
                rx: 10
            }
        },
        methods: {
		    onMouseDown: function(event) {
		        document.addEventListener('mousemove', this.onMouseMove);
		        document.addEventListener('mouseup', this.onMouseUp);
		        this.mouse.x = event.pageX;
		        this.mouse.y = event.pageY;
            },
            onMouseUp: function() {
                document.removeEventListener('mousemove', this.onMouseMove);
                document.removeEventListener('mouseup', this.onMouseUp);

                let workspace = document.getElementsByClassName('workspace')[0].getBoundingClientRect();
                if (this.tensor.position.x < this.width / 2) {
                    this.tensor.position.x = this.width / 2;
                }
                if (this.tensor.position.y < this.height / 2) {
                    this.tensor.position.y = this.height / 2;
                }
                if (this.tensor.position.x > workspace.width - this.width / 2) {
                    this.tensor.position.x = workspace.width - this.width / 2;
                }
                if (this.tensor.position.y > workspace.height - this.height / 2) {
                    this.tensor.position.y = workspace.height - this.height / 2;
                }
            },
            onMouseMove: function(event) {
                let dx = event.pageX - this.mouse.x;
                let dy = event.pageY - this.mouse.y;
                this.tensor.position.x += dx;
                this.tensor.position.y += dy;
                this.mouse.x = event.pageX;
                this.mouse.y = event.pageY;
            }
        },
		computed: {
			translation: function() {
				return 'translate(' + this.tensor.position.x + ' ' + this.tensor.position.y + ')';
			},
			style: function() {
				hue = Math.random() * 360;
				return 'fill: hsla(' + hue + ',100%,50%,0.3);'
			}
		},
		template: `
			<g class="tensor" :transform="translation" @mousedown="onMouseDown" @mouseup="onMouseUp">
				<rect :x="-width / 2" :y="-height / 2" :width="width" :height="height" :rx="rx" :style="style" />
				<text x="0" y="0">{{tensor.name}}</text>
			</g>
		`	
	}
);

Vue.component(
    'tensor-description',
    {
        mixins: [mixinTensor],
        template: `
            <p>Tensor {{tensor.name}} has {{tensor.axes.length}} axes:
                <ul>
                    <li v-for="(axisName, i) in tensor.axes">
                        Axis {{i}} <span v-if="axisName">({{axisName}})</span>
                        <span v-if="neighborAt(i)">is connected to axis {{neighborAt(i)[1]}}
                            <span v-if="getAxis(neighborAt(i))">({{getAxis(neighborAt(i))}})</span>
                            of tensor {{getTensor(neighborAt(i)[0]).name}}
                            <span v-if="edgeNameAt(i)">by edge "{{edgeNameAt(i)}}"</span>
                        </span>
                        <span v-else>is free</span>
                    </li>
                </ul>
            </p>
        `
    }
);
