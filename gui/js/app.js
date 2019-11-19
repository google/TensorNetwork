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

let app = new Vue({
    el: '#app',
    data: {
        state: initialState // now state object is reactive, whereas initialState is not
    },
    methods: {
        exportSVG: function(event) {
            event.preventDefault();
            let serializer = new XMLSerializer();
            let workspace = document.getElementById('workspace');
            let blob = new Blob([serializer.serializeToString(workspace)], {type:"image/svg+xml;charset=utf-8"});
            let url = URL.createObjectURL(blob);
            let link = document.createElement('a');
            link.href = url;
            link.download = "export.svg";
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    },
    template: `
        <div>
        <div class="app">
			<workspace :state="state" />
			<a href="" class="export" @click="exportSVG">Export SVG</a>
			<toolbar :state="state" />
        </div>
        <code-output :state="state" />
        </div>

    `
});
