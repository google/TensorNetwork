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
            tensor1: function() {
                return this.getTensor(this.edge[0][0]);
            },
            tensor2: function() {
                return this.getTensor(this.edge[1][0]);
            },
            angle1: function() {
                return this.axisAngle(this.edge[0][1], this.tensor1.axes.length);
            },
            angle2: function() {
                return this.axisAngle(this.edge[1][1], this.tensor2.axes.length)
            },
            x1: function() {
                return this.tensor1.position.x + this.axisX(this.angle1);
            },
            y1: function() {
                return this.tensor1.position.y + this.axisY(this.angle1);
            },
            x2: function() {
                return this.tensor2.position.x + this.axisX(this.angle2);
            },
            y2: function() {
                return this.tensor2.position.y + this.axisY(this.angle2);
            }
        },
        template: `
            <line class="edge" :x1="x1" :y1="y1" :x2="x2" :y2="y2"
                stroke="#ddd" stroke-width="5" stroke-linecap="round" />
        `
    }
);

