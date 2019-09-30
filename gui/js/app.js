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
		template: `
			<svg class="workspace" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="400" height="400">
				<tensor :tensor="tensor" :state="state" v-for="tensor in state.tensors" />
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
            <tensor-description :tensor="tensor" :state="state" v-for="tensor in state.tensors" />
			<workspace :state="state" />
        </div>
    `
});


