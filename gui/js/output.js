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
                let code = `import tensornetwork as tn\n`;

                code += `\n# Node definitions\n`;
                code += `# TODO: fill in ... values\n\n`;

                for (let i = 0; i < this.state.nodes.length; i++) {
                    let node = this.state.nodes[i];
                    code += `${node.name} = tn.Node(..., name="${node.name}")\n`;
                }

                code += `\n# Edge definitions\n\n`;

                for (let i = 0; i < this.state.edges.length; i++) {
                    let edge = this.state.edges[i];
                    code += `${edge[0][0]}[${edge[0][1]}] ^ ${edge[1][0]}[${edge[1][1]}]\n`;
                }

                return code;
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

