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

let mixinGet = {
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
	}
};

let mixinGeometry = {
	data: function() {
		return {
			axisLength: 50,
			tensorWidth: 50,
			tensorHeight: 50,
			tensorCornerRadius: 10
		}
	},
	methods: {
		axisAngle: function(index, nAxes) {
			return (index / nAxes) * 2 * Math.PI;
		},
		axisX: function(angle) {
			return this.axisLength * Math.cos(angle);
		},
		axisY: function(angle) {
			return this.axisLength * Math.sin(angle);
		}
	}
};