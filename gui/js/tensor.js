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
		mixins: [mixinGet, mixinGeometry, mixinTensor],
        data: function() {
		    return {
		        mouse: {
                    x: null,
                    y: null
                }
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
                if (this.tensor.position.x < this.tensorWidth / 2) {
                    this.tensor.position.x = this.tensorWidth / 2;
                }
                if (this.tensor.position.y < this.tensorHeight / 2) {
                    this.tensor.position.y = this.tensorHeight / 2;
                }
                if (this.tensor.position.x > workspace.width - this.tensorWidth / 2) {
                    this.tensor.position.x = workspace.width - this.tensorWidth / 2;
                }
                if (this.tensor.position.y > workspace.height - this.tensorHeight / 2) {
                    this.tensor.position.y = workspace.height - this.tensorHeight / 2;
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
				return 'fill: hsl(' + this.tensor.hue + ', 80%, 80%);'
			}
		},
        created: function() {
		    if (this.tensor.hue == null) {
		        this.tensor.hue = Math.random() * 360;
            }
        },
		template: `
			<g class="tensor" :transform="translation" @mousedown="onMouseDown" @mouseup="onMouseUp">
			    <axis v-for="(axisName, i) in tensor.axes" :tensor="tensor" :index="i" />
				<rect :x="-tensorWidth / 2" :y="-tensorHeight / 2" :width="tensorWidth"
				    :height="tensorHeight" :rx="tensorCornerRadius" :style="style" />
				<text x="0" y="0">{{tensor.name}}</text>
			</g>
		`	
	}
);

Vue.component(
    'axis',
    {
        mixins: [mixinGeometry],
        props: {
            tensor: Object,
            index: Number
        },
        methods: {
            onMouseDown: function(event) {
                event.stopPropagation();
            }
        },
        computed: {
            nAxes: function() {
                return this.tensor.axes.length;
            },
            angle: function() {
                return this.axisAngle(this.index, this.nAxes);
            },
            x: function() {
                return this.axisX(this.angle);
            },
            y: function() {
                return this.axisY(this.angle);
            },
            stroke: function() {
                return 'hsl(' + this.tensor.hue + ', 80%, 80%)'
            }
        },
        template: `
            <line x1="0" y1="0" :x2="x" :y2="y" :stroke="stroke" stroke-width="5" 
                stroke-linecap="round" @mousedown="onMouseDown"/>
        `
    }
);

Vue.component(
    'tensor-description',
    {
        mixins: [mixinGet, mixinTensor],
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
