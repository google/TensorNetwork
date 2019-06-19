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
import tensornetwork as tn
import experiments.MPS.misc_mps as misc_mps
import experiments.MPS.matrixproductstates  as MPS
import experiments.MPS.DMRG  as DMRG
import experiments.MPS.matrixproductoperators  as MPO
import pytest
import numpy as np

tf.enable_v2_behavior()

@pytest.mark.parametrize("direction", ['l', 'r'])
def test_TMeigs_float(direction):
    D, d, dtype = 10, 2, tf.float64
    tensors = [tf.random_uniform(shape=[D, d, D], dtype=dtype)]
    eta, x = misc_mps.TMeigs(tensors=tensors,
                              direction=direction)
    out = misc_mps.transfer_op(tensors,tensors,direction,x)
    np.testing.assert_allclose(out, eta * x)


@pytest.mark.parametrize("direction", ['l', 'r'])
def test_TMeigs_complex(direction):
    D, d, dtype = 10, 2, tf.float64
    tensors = [tf.complex(tf.random_uniform(shape=[D, d, D], dtype=dtype),tf.random_uniform(shape=[D, d, D], dtype=dtype))]
    eta, x = misc_mps.TMeigs(tensors=tensors,
                              direction=direction)
    out = misc_mps.transfer_op(tensors,tensors,direction,x)
    np.testing.assert_allclose(out, eta * x)

@pytest.mark.parametrize("direction", ['l', 'r'])
def test_TMeigs_pm_float(direction):
    D, d, dtype = 10, 2, tf.float64
    tensors = [tf.random_uniform(shape=[D, d, D], dtype=dtype)]
    eta, x, _, _ = misc_mps.TMeigs_power_method(tensors=tensors,
                              direction=direction)
    out = misc_mps.transfer_op(tensors,tensors,direction,x)
    np.testing.assert_allclose(out, eta * x)


@pytest.mark.parametrize("direction", ['l', 'r'])
def test_TMeigs_pm_complex(direction):
    D, d, dtype = 10, 2, tf.float64
    tensors = [tf.complex(tf.random_uniform(shape=[D, d, D], dtype=dtype),tf.random_uniform(shape=[D, d, D], dtype=dtype))]
    eta, x, _, _ = misc_mps.TMeigs_power_method(tensors=tensors,
                              direction=direction)
    out = misc_mps.transfer_op(tensors,tensors,direction,x)
    np.testing.assert_allclose(out, eta * x)

@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])    
def test_iDMRG(dtype):
    N, D, d = 2, 10, 2
    imps = MPS.InfiniteMPSCentralGauge.random(d=[2]*N,D=[D]*(N+1),dtype=dtype,minval=-0.5,maxval=0.5)
    impo = MPO.InfiniteXXZ(Jz=1.0*np.ones([N]),Jxy=np.ones([N]),Bz=np.zeros([N]),dtype=dtype)
    idmrg = DMRG.InfiniteDMRGEngine(imps,impo)
    e = idmrg.run_one_site(Nsweeps=100,verbose=1,ncv=100)
    assert(np.abs(-np.log(2) + 0.25 - e) < 5E-4)

@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])        
def test_DMRG(dtype):
    N, D, d = 10, 32, 2
    mps = MPS.FiniteMPSCentralGauge.random([d]*N,[D]*(N-1),dtype=dtype)
    mps.position(0)
    mps.position(len(mps))
    mps.position(0)
    mps.normalize()
    mpo = MPO.FiniteXXZ(Jz=np.ones([N-1]),Jxy=np.ones([N-1]),Bz=np.zeros([N]),dtype=dtype)
    dmrg = DMRG.FiniteDMRGEngine(mps,mpo)
    e = dmrg.run_one_site(verbose=1,Nsweeps=10,ncv=10,delta=1E-10)
    Eexact = -4.2580352072
    assert(np.abs(Eexact - e) < 5E-4)
    
@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])
def test_mps_position(dtype):
    N, D, d = 4, 10, 2
    imps = MPS.InfiniteMPSCentralGauge.random(d=[d]*N, D=[D]*(N+1), dtype=dtype, minval=-0.5, maxval=0.5)
    imps.position(len(imps))
    np.testing.assert_allclose([imps.ortho_deviation(imps.get_tensor(n),'l').numpy() for n in range(len(imps))], 0.0, atol=1E-9)
    imps.position(0)
    imps.position(len(imps))
    np.testing.assert_allclose([imps.ortho_deviation(imps.get_tensor(n),'l').numpy() for n in range(len(imps))], 0.0, atol=1E-9)
    
@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])    
def test_restore_form(dtype):
    N, D, d = 4, 10, 2
    imps = MPS.InfiniteMPSCentralGauge.random(d=[d]*N, D=[D]*(N+1), dtype=dtype, minval=-0.5, maxval=0.5)
    imps.restore_form()
    np.testing.assert_allclose([imps.ortho_deviation(imps.get_tensor(n),'l').numpy() for n in range(len(imps))], 0.0, atol=1E-9)
    imps.position(0)
    imps.position(len(imps))
    np.testing.assert_allclose([imps.ortho_deviation(imps.get_tensor(n),'l').numpy() for n in range(len(imps))], 0.0, atol=1E-9)

@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])    
def test_get_right_orthogonal_mps(dtype):
    N, D, d = 4, 10, 2    
    imps = MPS.InfiniteMPSCentralGauge.random(d=[d]*N, D=[D]*(N+1), dtype=dtype, minval=-0.5, maxval=0.5)    
    rimps=imps.get_right_orthogonal_imps()
    np.testing.assert_allclose([rimps.ortho_deviation(rimps.get_tensor(n),'r').numpy() for n in range(len(rimps))], 0.0, atol=1E-9)

@pytest.mark.parametrize("dtype", [tf.float64, tf.complex128])        
def test_get_left_orthogonal_mps(dtype):
    N, D, d = 4, 10, 2    
    imps = MPS.InfiniteMPSCentralGauge.random(d=[d]*N, D=[D]*(N+1), dtype=dtype, minval=-0.5, maxval=0.5)    
    limps=imps.get_left_orthogonal_imps()
    print('check orthogonality conditions (all numbers should be small)')
    np.testing.assert_allclose([limps.ortho_deviation(limps.get_tensor(n),'l').numpy() for n in range(len(limps))], 0.0, atol=1E-9)    

