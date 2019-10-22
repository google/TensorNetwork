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
    'code-output',
	{
        props: {
            state: Object
        },
        computed: {
            outputCode: function() {
                let code = `import numpy as np\nimport tensornetwork as tn\n`;

                code += `\n# Node definitions\n`;
                code += `# TODO: replace np.zeros with actual values\n\n`;

                for (let i = 0; i < this.state.nodes.length; i++) {
                    let node = this.state.nodes[i];
                    let values = this.placeholderValues(node);
                    let axes = this.axisNames(node);
                    code += `${node.name} = tn.Node(${values}, name="${node.name}"${axes})\n`;
                }

                code += `\n# Edge definitions\n\n`;

                for (let i = 0; i < this.state.edges.length; i++) {
                    let edge = this.state.edges[i];
                    let name = this.edgeName(edge);
                    code += `tn.connect(${edge[0][0]}[${edge[0][1]}], ${edge[1][0]}[${edge[1][1]}]${name})\n`;
                }

                return code;
            }
        },
        methods: {
            placeholderValues: function(node) {
                let code = `np.zeros((`;
                for (let i = 0; i < node.axes.length; i++) {
                    code += `0, `;
                }
                code += `))`;
                return code;
            },
            axisNames: function(node) {
                let code = `, axis_names=[`;
                let willOutput = false;
                for (let i = 0; i < node.axes.length; i++) {
                    let axis = node.axes[i];
                    if (axis) {
                        willOutput = true;
                        code += `"${axis}", `
                    }
                    else {
                        code += `None, `
                    }
                }
                code += `]`;
                return willOutput ? code : ``;
            },
            edgeName: function(edge) {
                let name = edge[2];
                return name ? `, name="${name}"` : ``;
            }
        },
		template: `
			<div class="code-output">
                <h2>TensorNetwork Output</h2>
                <pre>{{outputCode}}</pre>
			</div>
		`
	}
);

