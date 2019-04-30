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


"""implementation of different Matrix Product Operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('../')
import ncon as ncon
import numpy as np
import copy
import tensorflow as tf

def kron(a,b):
    a1,a2=a.shape
    b1,b2=b.shape
    return tf.reshape(tf.transpose(tf.tensordot(a,b,axes=0),(0,2,1,3)),(a1*b1,a2*b2))

class MPOBase:

    def __init__(self, tensors, name=None):
        self.name = None
        self._tensors = tensors
        assert (np.all([self[0].dtype == m.dtype for m in self]))

    def __getitem__(self, n):
        return self._tensors[n]

    def __setitem__(self, n):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._tensors)

    def __len__(self):
        return len(self._tensors)

    @property
    def dtype(self):
        assert (np.all([self[0].dtype == m.dtype for m in self]))
        return self._tensors[0].dtype

    @property
    def D(self):
        """Returns a vector of all bond dimensions.
        The vector will have length `N+1`, where `N == num_sites`."""
        return ([self.get_tensor(0).shape[0]] +
                [self.get_tensor(n).shape[1] for n in range(len(self))])

    def get_tensor(self, site):
        return self._tensors[site]

    def set_tensor(self, site, tensors):
        raise NotImplementedError()

    def get_2site_mpo(self, *args, **kwargs):
        raise NotImplementedError()
    def get_2site_hamiltonian(self, *args, **kwargs):
        raise NotImplementedError()
    
    def get_2site_gate(self, site1, site2, tau):
        tau=tf.convert_to_tensor(tau)
        """
        calculate the unitary two-site gates exp(tau*H(m,n))
        Parameters:
        --------------------------------------
        site1,site2: int
                     lattice sites for which to calculate the gate
        tau:         float or complex
                     time-increment
        Returns:
        --------------------------------------------------
        A two-site gate "Gate" between sites m and n by summing up  (morally, for m<n)
        h=\sum_s kron(mpo[m][-1,s,:,:],mpo[n][s,0,:,:]) and exponentiating the result:
        Gate=scipy.linalg..expm(tau*h); 
        Gate is a rank-4 tensor with shape (dm,dn,dm,dn), with
        dm, dn the local hilbert space dimension at site m and n, respectively
        """
        if site2 < site1:
            d1 = self[site2].shape[2]
            d2 = self[site1].shape[2]
        elif site2 > site1:
            d1 = self[site1].shape[2]
            d2 = self[site2].shape[2]
        else:
            raise ValuError(
                'MPOBase.get_2site_gate: site1 has to be different from site2!')
        h = tf.reshape(
            self.get_2site_hamiltonian(site1, site2), (d1 * d2, d1 * d2))
        if not h.dtype==tau.dtype:
            raise TypeError('MPOBase.get_2site_gate: expected tau of dtype {0}, got dtype {1}'.format(self.dtype,tau.dtype))
        
        return tf.reshape(tf.linalg.expm(tau * h), (d1, d2, d1, d2))

class InfiniteMPO(MPOBase):
    def __init__(self, tensors, name=None):
        super().__init__(tensors=tensors, name=name)
        if not (self.D[0] == self.D[-1]):
            raise ValueError('InfiniteMPO: left and right MPO ancillary dimension do not match')
        
    def get_boundary_vector(self, side):
        if side.lower() in ('l', 'left'):
            v = tf.zeros([self.D[0]], dtype=self.dtype)
            v.numpy()[-1] = 1.0
            return v

        if side.lower() in ('r', 'right'):
            v = tf.zeros([self.D[-1]], dtype=self.dtype)
            v.numpy()[0] = 1.0
            return v

    def get_boundary_mpo(self, side):
        if side.lower() in ('l', 'left'):
            out = copy.deepcopy(self._tensors[-1][-1, :, :, :])
            out.numpy()[0, :, :] *= 0.0
        if side.lower() in ('r', 'right'):
            out = copy.deepcopy(self._tensors[0][:, 0, :, :])
            out.numpy()[-1, :, :] *= 0.0
        return tf.squeeze(out)
    
    def get_2site_mpo(self, site1, site2):
        if site2 < site1:
            mpo1 = copy.deepcopy(self[site2][-1, :, :, :])
            mpo2 = copy.deepcopy(self[site1][:, 0, :, :])
            if site2 == 0:
                mpo1[0, :, :] /= 2.0
            if site1 == (len(self) - 1):
                mpo2[-1, :, :] /= 2.0

        if site2 > site1:
            mpo1 = copy.deepcopy(self[site1][-1, :, :, :])
            mpo2 = copy.deepcopy(self[site2][:, 0, :, :])
            if site1 == 0:
                mpo1[0, :, :] /= 2.0
            if site2 == (len(self) - 1):
                mpo2[-1, :, :] /= 2.0

        d1 = mpo1.shape[1]
        d2 = mpo2.shape[1]

        return [
            tf.expand_dims(mpo1, 0),
            tf.expand_dims(mpo2, 1)
        ]


    def get_2site_hamiltonian(self, site1, site2):
        """
        obtain a two-site Hamiltonian H_{mn} from MPO
        Parameters:
        --------------------------------------
        site1,site2: int
                     lattice sites for which to calculate the Hamiltonian
        Returns:
        --------------------------------------------------
        tf.Tensor of shape (d1,d2,d3,d4)
        A two-site Hamiltonian between sites ```site1``` and ```site2``` by summing up  
        (for site1<site2, and site1!=0, site2!=0)

        \sum_s={0}^{M-1} kron(mpo[m][-1,s,:,:],mpo[n][s,0,:,:])

        the returned tf.Tensor is a rank-4 tensor with shape (dsite1,dsite2,dsite1,dsite2), with
        dsite1, dsite2 the local hilbert space dimension at sites ```site1``` and ```site2```, respectively,
        
        """
        mpo1, mpo2 = self.get_2site_mpo(site1, site2)
        if site2 < site1:
            nl = site2
            mr = site1
        elif site2 > site1:
            nl = site1
            nr = site2
        mpo1 = mpo1[0, :, :, :]
        mpo2 = mpo2[:, 0, :, :]
            
        d1 = mpo1.shape[1]
        d2 = mpo2.shape[1]

        h=kron(mpo1[0,:,:],mpo2[0,:,:])
        for s in range(1,mpo1.shape[0]):
            h+=kron(mpo1[s,:,:],mpo2[s,:,:])

        return tf.reshape(h, (d1, d2, d1, d2))

    def roll(self,num_sites):
        tensors=[self._tensors[n] for n in range(num_sites,len(self._tensors))]\
            + [self._tensors[n] for n in range(num_sites)]
        self._tensors=tensors



class FiniteMPO(MPOBase):
    def __init__(self, tensors, name=None):
        super().__init__(tensors=tensors, name=name)
        if not (self.D[0] == 1) and (self.D[-1] == 1):
            raise ValueError('FiniteMPO: left and right MPO ancillary dimension is different from 1')

    def get_2site_mpo(self, site1, site2):
        if site2 < site1:
            mpo1 = self[site2][-1, :, :, :]
            mpo2 = self[site1][:, 0, :, :]

        if site2 > site1:
            mpo1 = self[site1][-1, :, :, :]
            mpo2 = self[site2][:, 0, :, :]

        d1 = mpo1.shape[1]
        d2 = mpo2.shape[1]

        return [
            tf.expand_dims(mpo1, 0),
            tf.expand_dims(mpo2, 1)
        ]

    def get_2site_hamiltonian(self, site1, site2):
        """
        obtain a two-site Hamiltonian H_{mn} from MPO
        Parameters:
        --------------------------------------
        site1,site2: int
                     lattice sites for which to calculate the Hamiltonian
        Returns:
        --------------------------------------------------
        tf.Tensor of shape (d1,d2,d3,d4)
        A two-site Hamiltonian between sites ```site1``` and ```site2``` by summing up  
        (for site1<site2, and site1!=0, site2!=0)
        h=kron(mpo[m][-1,s=0,:,:]/2,mpo[n][s=0,0,:,:])+
          \sum_s={1}^{M-2} kron(mpo[m][-1,s,:,:],mpo[n][s,0,:,:])+
          kron(mpo[m][-1,s=M-1,:,:],mpo[n][s=M-1,0,:,:])+
        the returned tf.Tensor is a rank-4 tensor with shape (dsite1,dsite2,dsite1,dsite2), with
        dsite1, dsite2 the local hilbert space dimension at sites ```site1``` and ```site2```, respectively,
        
        """
        mpo1, mpo2 = self.get_2site_mpo(site1, site2)
        if site2 < site1:
            nl = site2
            mr = site1
        elif site2 > site1:
            nl = site1
            nr = site2

        mpo1 = mpo1[0, :, :, :]
        mpo2 = mpo2[:, 0, :, :]
        d1 = mpo1.shape[1]
        d2 = mpo2.shape[1]
        if nl != 0 and nr != (len(self) - 1):
            h = kron(mpo1[0, :, :] / 2.0, mpo2[0, :, :])
            for s in range(1, mpo1.shape[0] - 1):
                h += kron(mpo1[s, :, :], mpo2[s, :, :])
            h += kron(mpo1[-1, :, :], mpo2[-1, :, :] / 2.0)

        elif nl != 0 and nr == (len(self) - 1):
            h = kron(mpo1[0, :, :] / 2.0, mpo2[0, :, :])
            for s in range(1, mpo1.shape[0]):
                h += kron(mpo1[s, :, :], mpo2[s, :, :])

        elif nl == 0 and nr != (len(self) - 1):
            h = kron(mpo1[0, :, :], mpo2[0, :, :])
            for s in range(1, mpo1.shape[0] - 1):
                h += kron(mpo1[s, :, :], mpo2[s, :, :])
            h += kron(mpo1[-1, :, :], mpo2[-1, :, :] / 2.0)

        elif nl == 0 and nr == (len(self) - 1):
            h = kron(mpo1[0, :, :], mpo2[0, :, :])
            for s in range(1, mpo1.shape[0]):
                h += kron(mpo1[s, :, :], mpo2[s, :, :])
        return tf.reshape(h, (d1, d2, d1, d2))

        
class FiniteXXZ(FiniteMPO):
    """
    the famous Heisenberg Hamiltonian, the one we all know and love (almost as much as TFI)!
    """

    def __init__(self, Jz, Jxy, Bz, dtype):
        
        if hasattr(dtype, 'as_numpy_dtype'):
           dtype = dtype.as_numpy_dtype
        self.Jz = Jz
        self.Jxy = Jxy
        self.Bz = Bz
        N = len(Bz)
        mpo = []
        temp = np.zeros((1, 5, 2, 2), dtype=dtype)
        #BSz
        temp[0, 0, 0, 0] = -0.5 * Bz[0]
        temp[0, 0, 1, 1] = 0.5 * Bz[0]

        #Sm
        temp[0, 1, 0, 1] = Jxy[0] / 2.0 * 1.0
        #Sp
        temp[0, 2, 1, 0] = Jxy[0] / 2.0 * 1.0
        #Sz
        temp[0, 3, 0, 0] = Jz[0] * (-0.5)
        temp[0, 3, 1, 1] = Jz[0] * 0.5

        #11
        temp[0, 4, 0, 0] = 1.0
        temp[0, 4, 1, 1] = 1.0
        mpo.append(tf.convert_to_tensor(temp))
        for n in range(1, N - 1):
            temp = np.zeros((5, 5, 2, 2), dtype=dtype)
            #11
            temp[0, 0, 0, 0] = 1.0
            temp[0, 0, 1, 1] = 1.0
            #Sp
            temp[1, 0, 1, 0] = 1.0
            #Sm
            temp[2, 0, 0, 1] = 1.0
            #Sz
            temp[3, 0, 0, 0] = -0.5
            temp[3, 0, 1, 1] = 0.5
            #BSz
            temp[4, 0, 0, 0] = -0.5 * Bz[n]
            temp[4, 0, 1, 1] = 0.5 * Bz[n]

            #Sm
            temp[4, 1, 0, 1] = Jxy[n] / 2.0 * 1.0
            #Sp
            temp[4, 2, 1, 0] = Jxy[n] / 2.0 * 1.0
            #Sz
            temp[4, 3, 0, 0] = Jz[n] * (-0.5)
            temp[4, 3, 1, 1] = Jz[n] * 0.5
            #11
            temp[4, 4, 0, 0] = 1.0
            temp[4, 4, 1, 1] = 1.0

            mpo.append(tf.convert_to_tensor(temp))
        temp = np.zeros((5, 1, 2, 2), dtype=dtype)
        #11
        temp[0, 0, 0, 0] = 1.0
        temp[0, 0, 1, 1] = 1.0
        #Sp
        temp[1, 0, 1, 0] = 1.0
        #Sm
        temp[2, 0, 0, 1] = 1.0
        #Sz
        temp[3, 0, 0, 0] = -0.5
        temp[3, 0, 1, 1] = 0.5
        #BSz
        temp[4, 0, 0, 0] = -0.5 * Bz[-1]
        temp[4, 0, 1, 1] = 0.5 * Bz[-1]

        mpo.append(tf.convert_to_tensor(temp))
        super().__init__(mpo, name='XXZ_MPO')


        
class InfiniteXXZ(InfiniteMPO):
    """
    the famous Heisenberg Hamiltonian, the one we all know and love (almost as much as TFI)!
    """

    def __init__(self, Jz, Jxy, Bz, dtype):

        if hasattr(dtype, 'as_numpy_dtype'):
           dtype = dtype.as_numpy_dtype
        
        self.Jz = Jz
        self.Jxy = Jxy
        self.Bz = Bz
        N = len(Bz)
        if not len(Jz) == len(Jxy):
            raise TypeError(
                'XXZ: Jz and Jxz have to be of same lengths for pbc')
        if not len(Bz) == len(Jxy):
            raise TypeError(
                'XXZ: Bz and Jxz have to be of same lengths for pbc')

        mpo = []
        for n in range(0, N):

            temp = np.zeros((5, 5, 2, 2), dtype=dtype)
            #11
            temp[0, 0, 0, 0] = 1.0
            temp[0, 0, 1, 1] = 1.0
            #Sp
            temp[1, 0, 1, 0] = 1.0
            #Sm
            temp[2, 0, 0, 1] = 1.0
            #Sz
            temp[3, 0, 0, 0] = -0.5
            temp[3, 0, 1, 1] = 0.5
            #BSz
            temp[4, 0, 0, 0] = -0.5 * Bz[n]
            temp[4, 0, 1, 1] = 0.5 * Bz[n]

            #Sm
            temp[4, 1, 0, 1] = Jxy[n] / 2.0 * 1.0
            #Sp
            temp[4, 2, 1, 0] = Jxy[n] / 2.0 * 1.0
            #Sz
            temp[4, 3, 0, 0] = Jz[n] * (-0.5)
            temp[4, 3, 1, 1] = Jz[n] * 0.5
            #11
            temp[4, 4, 0, 0] = 1.0
            temp[4, 4, 1, 1] = 1.0

            mpo.append(tf.convert_to_tensor(temp))
        super().__init__(mpo, name='XXZ_MPO')


class FiniteTFI(FiniteMPO):
    """
    the good old transverse field Ising model
    convention: sigma_z=diag([-1,1])
    """

    def __init__(self, Jx, Bz, dtype=tf.float64):
        
        if hasattr(dtype, 'as_numpy_dtype'):
           dtype = dtype.as_numpy_dtype
        
        self.Jx = Jx.astype(dtype)
        self.Bz = Bz.astype(dtype)
        N = len(Bz)
        sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
        sigma_z = np.diag([-1, 1]).astype(dtype)
        mpo = []
        temp = np.zeros(shape=[1, 3, 2, 2], dtype=dtype)
        #Bsigma_z
        temp[0, 0, :, :] = self.Bz[0] * sigma_z
        #sigma_x
        temp[0, 1, :, :] = self.Jx[0] * sigma_x
        #11
        temp[0, 2, 0, 0] = 1.0
        temp[0, 2, 1, 1] = 1.0
        mpo.append(tf.convert_to_tensor(temp))
        for n in range(1, N - 1):
            temp = np.zeros(shape=[3, 3, 2, 2], dtype=dtype)
            #11
            temp[0, 0, 0, 0] = 1.0
            temp[0, 0, 1, 1] = 1.0
            #sigma_x
            temp[1, 0, :, :] = sigma_x
            #Bsigma_z
            temp[2, 0, :, :] = self.Bz[n] * sigma_z
            #sigma_x
            temp[2, 1, :, :] = self.Jx[n] * sigma_x
            #11
            temp[2, 2, 0, 0] = 1.0
            temp[2, 2, 1, 1] = 1.0
            mpo.append(tf.convert_to_tensor(temp))

        temp = np.zeros([3, 1, 2, 2], dtype=dtype)
        #11
        temp[0, 0, 0, 0] = 1.0
        temp[0, 0, 1, 1] = 1.0
        #sigma_x
        temp[1, 0, :, :] = sigma_x
        #Bsigma_z
        temp[2, 0, :, :] = self.Bz[-1] * sigma_z
        mpo.append(tf.convert_to_tensor(temp))

        super().__init__(tensors=mpo, name='TFI_MPO')



        
class InfiniteTFI(InfiniteMPO):
    """
    the good old transverse field Ising model
    convention: sigma_z=diag([-1,1])
    """

    def __init__(self, Jx, Bz, dtype=tf.float64):
        if hasattr(dtype, 'as_numpy_dtype'):
           dtype = dtype.as_numpy_dtype
        
        self.Jx = Jx.astype(dtype)
        self.Bz = Bz.astype(dtype)
        N = len(Bz)
        sigma_x = np.array([[0, 1], [1, 0]]).astype(dtype)
        sigma_z = np.diag([-1, 1]).astype(dtype)
        mpo = []
        for n in range(0, N):
            temp = np.zeros(shape=[3, 3, 2, 2], dtype=dtype)

            #11
            temp[0, 0, 0, 0] = 1.0
            temp[0, 0, 1, 1] = 1.0

            #sigma_x

            temp[1, 0, 1, 0] = 1
            temp[1, 0, 0, 1] = 1

            #Bsigma_z
            temp[2, 0:, :] = sigma_z * self.Bz[n]

            #sigma_x
            temp[2, 1, :, :] = sigma_x * self.Jx[n]

            #11
            temp[2, 2, 0, 0] = 1.0
            temp[2, 2, 1, 1] = 1.0
            mpo.append(tf.convert_to_tensor(temp))
        super().__init__(tensors=mpo, name='TFI_MPO')
