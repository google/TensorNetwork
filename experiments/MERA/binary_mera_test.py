# Copyright 2019 The TensorNetwork Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
unittests
"""
import tensorflow as tf
import numpy as np
import tensornetwork as tn
import experiments.MERA.binary_mera_lib as bml
import experiments.MERA.binary_mera as bm
import experiments.MERA.misc_mera as misc_mera
import pytest
import copy
tn.set_default_backend("tensorflow")
tf.enable_v2_behavior()

@pytest.mark.parametrize("chi", [4, 6])
@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])
def test_ascending_descending(chi, dtype):
    """
    test if ascending and descending operations are doing the right thing
    """
    wC, uC, rho_0 = bml.initialize_binary_MERA_random(phys_dim=2, chi=chi, dtype=dtype)
    wC, uC = bml.unlock_layer(wC, uC) #add a transitional layer
    wC, uC = bml.unlock_layer(wC, uC) #add a transitional layer
    ham_0 = bml.initialize_TFI_hams(dtype)
    rho = [0 for n in range(len(wC) + 1)]
    ham = [0 for n in range(len(wC) + 1)]
    rho[-1] = bml.steady_state_density_matrix(10, rho_0, wC[-1], uC[-1])
    ham[0] = ham_0
    for p in range(len(rho) - 2, -1, -1):
      rho[p] = bml.descending_super_operator(rho[p + 1], wC[p], uC[p])
    for p in range(len(wC)):
      ham[p + 1] = bml.ascending_super_operator(ham[p], wC[p], uC[p])
    energies = [
        tn.ncon([rho[p], ham[p]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
        for p in range(len(rho))
    ]
    np.testing.assert_allclose(
        np.array(
            [energies[p] / energies[p + 1] for p in range(len(energies) - 1)]),
        0.5)


@pytest.mark.parametrize("chi", [4, 6])
@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])
def test_steady_state(chi, dtype):
    isometry = misc_mera.w_update_svd_numpy(
        np.random.rand(chi, chi, chi).astype(dtype.as_numpy_dtype))
    unitary = misc_mera.u_update_svd_numpy(
        np.random.rand(chi, chi, chi, chi).astype(dtype.as_numpy_dtype))
    rho = tf.reshape(
        tf.eye(chi * chi * chi, dtype=dtype), (chi, chi, chi, chi, chi, chi))
    rho_ss = bml.steady_state_density_matrix(
        nsteps=60, rho=rho, isometry=isometry, unitary=unitary)
    rho_test = bml.descending_super_operator(rho_ss, isometry, unitary)
    np.testing.assert_array_less(rho_ss - rho_test, 1E-6)


@pytest.mark.parametrize("chi", [4, 6])
@pytest.mark.parametrize("dtype", [tf.float64])
def test_disentangler_envs(chi, dtype):
    isometry = misc_mera.w_update_svd_numpy(
        np.random.rand(chi, chi, chi).astype(dtype.as_numpy_dtype))
    unitary = misc_mera.u_update_svd_numpy(
        np.random.rand(chi, chi, chi, chi).astype(dtype.as_numpy_dtype))
    rho = tf.random_uniform(shape=[chi, chi, chi, chi, chi, chi], dtype=dtype)
    ham = tf.random_uniform(shape=[chi, chi, chi, chi, chi, chi], dtype=dtype)
    
    envs = {}
    envs[0] = bml.get_env_disentangler_1(ham, rho, isometry, unitary)
    envs[1] = bml.get_env_disentangler_3(ham, rho, isometry, unitary)
    envs[2] = bml.get_env_disentangler_2(ham, rho, isometry, unitary)
    envs[3] = bml.get_env_disentangler_4(ham, rho, isometry, unitary)
    
    t_a = [
        tn.ncon([envs[n], unitary], [[1, 2, 3, 4], [1, 2, 3, 4]])
        for n in range(2)
    ]
    t_b = [
        tn.ncon([envs[n], unitary], [[1, 2, 3, 4], [1, 2, 3, 4]])
        for n in range(2, 4)
    ]
    np.testing.assert_allclose(t_a, t_a[0])
    np.testing.assert_allclose(t_b, t_b[0])


@pytest.mark.parametrize("chi", [4, 6])
@pytest.mark.parametrize("dtype", [tf.float64])
def test_isometry_envs(chi, dtype):
    isometry = misc_mera.w_update_svd_numpy(
        np.random.rand(chi, chi, chi).astype(dtype.as_numpy_dtype))
    unitary = misc_mera.u_update_svd_numpy(
        np.random.rand(chi, chi, chi, chi).astype(dtype.as_numpy_dtype))
    rho = tf.random_uniform(shape=[chi, chi, chi, chi, chi, chi], dtype=dtype)
    ham = tf.random_uniform(shape=[chi, chi, chi, chi, chi, chi], dtype=dtype)
    
    envs = {}
    envs[0] = bml.get_env_isometry_1(ham, rho, isometry, unitary)
    envs[1] = bml.get_env_isometry_3(ham, rho, isometry, unitary)
    envs[2] = bml.get_env_isometry_5(ham, rho, isometry, unitary)
    envs[3] = bml.get_env_isometry_2(ham, rho, isometry, unitary)
    envs[4] = bml.get_env_isometry_4(ham, rho, isometry, unitary)
    envs[5] = bml.get_env_isometry_6(ham, rho, isometry, unitary)
    t_a = [tn.ncon([envs[n], isometry], [[1, 2, 3], [1, 2, 3]]) for n in range(3)]
    t_b = [
        tn.ncon([envs[n], isometry], [[1, 2, 3], [1, 2, 3]]) for n in range(3, 6)
    ]
    np.testing.assert_allclose(t_a, t_a[0])
    np.testing.assert_allclose(t_b, t_b[0])

@pytest.mark.parametrize("chi", [4, 6])
@pytest.mark.parametrize("dtype", [tf.float64])
def test_padding(chi, dtype):
    wC, uC, rho_0 = bml.initialize_binary_MERA_random(phys_dim=2, chi=chi, dtype=dtype)
    wC, uC = bml.unlock_layer(wC, uC) #add a transitional layer
    wC, uC = bml.unlock_layer(wC, uC) #add a transitional layer
    ham_0 = bml.initialize_TFI_hams(dtype)
    def get_energies(wC, uC, rho_0, ham_0):
        rho = [0 for n in range(len(wC) + 1)]
        ham = [0 for n in range(len(wC) + 1)]
      
        rho[-1] = bml.steady_state_density_matrix(10, rho_0, wC[-1], uC[-1])
        ham[0] = ham_0
        for p in range(len(rho) - 2, -1, -1):
          rho[p] = bml.descending_super_operator(rho[p + 1], wC[p], uC[p])
        for p in range(len(wC)):
          ham[p + 1] = bml.ascending_super_operator(ham[p], wC[p], uC[p])
        energies = [
          tn.ncon([rho[p], ham[p]], [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])
          for p in range(len(rho))
        ]
        return energies

    energies_1 = get_energies(wC, uC, rho_0, ham_0)
    
    chi_new = chi + 1
    wC, uC = bml.pad_mera_tensors(chi_new, wC, uC)
    rho_0 = misc_mera.pad_tensor(rho_0, [chi_new] * 6)
    energies_2 = get_energies(wC, uC, rho_0, ham_0)
    np.testing.assert_allclose(energies_1, energies_2)

