import sys
from sys import stdout
import tensorflow as tf
import copy
import numpy as np
import itertools
import math
import experiments.MPS.misc_mps as misc_mps
import tensornetwork as tn
import time
import pickle
tf.enable_eager_execution()
tf.enable_v2_behavior()


def svd(mat, full_matrices=False, compute_uv=True, r_thresh=1E-12):
    """
    wrapper around numpy svd
    catches a weird LinAlgError exception sometimes thrown by lapack (cause not entirely clear, but seems 
    related to very small matrix entries)
    if LinAlgError is raised, precondition with a QR decomposition (fixes the problem)


    Parameters
    ----------
    mat:           array_like of shape (M,N)
                   A real or complex array with ``a.ndim = 2``.
    full_matrices: bool, optional
                   If True (default), `u` and `vh` have the shapes ``(M, M)`` and
                   (N, N)``, respectively.  Otherwise, the shapes are
                   (M, K)`` and ``(K, N)``, respectively, where
                   K = min(M, N)``.
    compute_uv :   bool, optional
                   Whether or not to compute `u` and `vh` in addition to `s`.  True
                   by default.

    Returns
    -------
    u : { (M, M), (M, K) } array
        Unitary array(s). The shape depends
        on the value of `full_matrices`. Only returned when
        `compute_uv` is True.
    s : (..., K) array
        Vector(s) with the singular values, within each vector sorted in
        descending order. The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`.
    vh : { (..., N, N), (..., K, N) } array
        Unitary array(s). The first ``a.ndim - 2`` dimensions have the same
        size as those of the input `a`. The size of the last two dimensions
        depends on the value of `full_matrices`. Only returned when
        `compute_uv` is True.
    """
    try:
        [u, s, v] = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.linalg.LinAlgError:
        [q, r] = np.linalg.qr(mat)
        r[np.abs(r) < r_thresh] = 0.0
        u_, s, v = np.linalg.svd(r)
        u = q.dot(u_)
        print('caught a LinAlgError with dir>0')
    return u, s, v

def split_node_full_svd_numpy(node, left_edges, right_edges, direction, max_singular_values=None, trunc_thresh=None):

    node = node.reorder_edges(left_edges + right_edges)
    D0, D1, D2, D3 = node.tensor.get_shape()    
    arr2 = np.reshape(node.tensor, [D0*D1, D2*D3])
    u,s,v = svd(arr2,full_matrices=False)
    if trunc_thresh:
        s /= np.linalg.norm(s)    
        s = s[s >= trunc_thresh]
    if max_singular_values:
        if len(s)> max_singular_values:
            s = s[0 : max_singular_values]
            
    s /= np.linalg.norm(s)
    u = u[:, 0 : len(s)]
    v = v[0 : len(s), :]
    if direction in  ('r','right'):
        out = np.transpose(np.reshape(u,(D0, D1, len(s))), [0, 2, 1])
        return tf.convert_to_tensor(out),tf.convert_to_tensor(np.reshape(np.dot(np.diag(s),v),(len(s), D2, D3)))
    if direction in  ('l','left'):
        return tf.convert_to_tensor(np.reshape(np.dot(u,np.diag(s)),(D0, D1, len(s)))),\
            tf.convert_to_tensor(np.reshape(v,(len(s), D2, D3)))

    
@tf.contrib.eager.defun
def prepare_tensor_SVD(tensor, direction, D=None, thresh=1E-32):
    """
    prepares and truncates an mps tensor using svd
    Parameters:
    ---------------------
    tensor: np.ndarray of shape(D1,D2,d)
            an mps tensor
    direction: int
               if >0 returns left orthogonal decomposition, if <0 returns right orthogonal decomposition
    thresh: float
            cutoff of schmidt-value truncation
    r_thresh: float
              only used when svd throws an exception.
    D:        int or None
              the maximum bond-dimension to keep (hard cutoff); if None, no truncation is applied

    Returns:
    ----------------------------
    direction>0: out,s,v,Z
                 out: a left isometric tensor of dimension (D1,D,d)
                 s  : the singular values of length D
                 v  : a right isometric matrix of dimension (D,D2)
                 Z  : the norm of tensor, i.e. tensor"="out.dot(s).dot(v)*Z
    direction<0: u,s,out,Z
                 u  : a left isometric matrix of dimension (D1,D)
                 s  : the singular values of length D
                 out: a right isometric tensor of dimension (D,D2,d)
                 Z  : the norm of tensor, i.e. tensor"="u.dot(s).dot(out)*Z

    """

    assert (direction != 0), 'do NOT use direction=0!'
    [l1, l2, d] = tensor.shape
    if direction in (1, 'l', 'left'):
        net = tn.TensorNetwork()
        node = net.add_node(tensor)
        u_node, s_node, v_node, _ = net.split_node_full_svd(node, [node[0], node[2]], [node[1]], max_singular_values=D, max_truncation_err=thresh)
        Z = tf.linalg.norm(s_node.tensor)
        s_node.tensor /= Z
        out = u_node.reorder_axes([0, 2, 1])
        return out.tensor, s_node.tensor, v_node.tensor, Z

    if direction in (-1, 'r', 'right'):
        net = tn.TensorNetwork()
        node = net.add_node(tensor)
        u_node, s_node, v_node, _ = net.split_node_full_svd(node, [node[0]], [node[1], node[2]], max_singular_values=D, max_truncation_err=thresh)
        Z = tf.linalg.norm(s_node.tensor)
        s_node.tensor /= Z
        return u_node.tensor, s_node.tensor, v_node.tensor, Z


    
class MPSClassifier:
    """
    A classifier for data
    Members:
        self.right_data_environments (dict): contains right data-environments.
                                             self.right_data_environments[site] contains the 
                                             right environment of site `site`
        self.left_data_environments (dict):  contains left data-environments.
                                             self.left_data_environments[site] contains the 
                                             left environment of site `site`
    """

    @classmethod
    def eye(cls, d, D, num_labels, dtype=tf.float64, noise=1E-5,scaling=0.1, name='MPS_classifier'):
        D = [1] + D
        tensors = [np.transpose(np.reshape(np.eye(D[n], D[n + 1])[list(range(D[n]))*d[n]],(d[n], 
                                                                                           D[n], D[n + 1])),
                               (1,2,0)).astype(dtype.as_numpy_dtype)
                   for n in range(len(d))]
        for t in tensors:
            t += (np.random.random_sample(t.shape)).astype(dtype.as_numpy_dtype) * noise
            t *= scaling
        label = (np.random.rand(D[-1], num_labels, 1).astype(dtype.as_numpy_dtype) + 0.5 )* noise   
        tf_tensors =  [tf.convert_to_tensor(t) for t in tensors]
        return cls(mps_tensors=tf_tensors, label_tensor=tf.convert_to_tensor(label), dtype=dtype, name=name)
    
    @classmethod
    def random(cls, d, D, num_labels, dtype=tf.float64, scaling=0.1, name='MPS_classifier'):
        D = [1] + D
        tensors = [np.random.randn(D[n], D[n + 1], d[n])*scaling
                   for n in range(len(d))]
        label = np.random.randn(D[-1], num_labels, 1)
        tf_tensors =  [tf.convert_to_tensor(t) for t in tensors]        
        return cls(mps_tensors=tf_tensors, label_tensor=tf.convert_to_tensor(label), dtype=dtype, name=name)    

    def __init__(self, mps_tensors, label_tensor, dtype, name = 'MPS_classifier'):
        self.Nlabels = label_tensor.shape[1]
        self._position = len(mps_tensors)
        self.label_tensor = label_tensor
        self.right_data_environment={}
        self.left_data_environment={}
        self.name = name
        self.right_envs = {}
        self.tensors=mps_tensors
        self.dtype=dtype
        
    def __len__(self):
        return len(self.tensors)

          

    @staticmethod
    #@tf.contrib.eager.defun    
    def split_off(tensor, direction, numpy_svd=False, D=None, trunc_thresh=None):
        if direction in ('r','right'):
            net = tn.TensorNetwork()
            t_node = net.add_node(tensor)
            left_edges = [t_node[0],  t_node[3]]
            right_edges = [t_node[1], t_node[2]]        

            if not numpy_svd:
                u_node, s_node, v_node, _ = net.split_node_full_svd(t_node, left_edges, right_edges, max_singular_values=D, max_truncation_err=trunc_thresh)
                Z = tf.linalg.norm(s_node.tensor)
                s_node.tensor /= Z
                out = u_node.reorder_axes([0, 2, 1])
                label_tensor = net.contract(s_node[1])
                return out.tensor , label_tensor.tensor
            else:
                out, label_tensor = split_node_full_svd_numpy(t_node, left_edges, right_edges, direction='r', max_singular_values=D, trunc_thresh=trunc_thresh)
                #print('out.shape: ',out.shape, 'label.shape: ', label_tensor.shape)            
                return out, label_tensor

        if direction in ('l','left'):            
            net = tn.TensorNetwork()
            t_node = net.add_node(tensor)
            left_edges = [t_node[0],  t_node[1]]
            right_edges = [t_node[2], t_node[3]]
            if not numpy_svd:
                u_node, s_node, v_node, _ = net.split_node_full_svd(t_node, left_edges, right_edges, max_singular_values=D, max_truncation_err=trunc_thresh)
                Z = tf.linalg.norm(s_node.tensor)
                s_node.tensor /= Z
                label_tensor = net.contract(s_node[0])
                return label_tensor.tensor, v_node.tensor
            else:
                #the numpy truncation scheme is different from tensorflow above
                label_tensor, out = split_node_full_svd_numpy(t_node, left_edges, right_edges, direction='l', max_singular_values=D,
                                                              trunc_thresh=trunc_thresh)
                return label_tensor, out
                
    @staticmethod        
    def shift_right(label_tensor, tensor, numpy_svd=False, D=None, trunc_thresh=None):
        t = misc_mps.ncon([label_tensor, tensor],[[-1,-2,1], [1,-3,-4]])        
        return MPSClassifier.split_off(t, direction='r', numpy_svd=numpy_svd, D=D, trunc_thresh=trunc_thresh)    

        

    
    @staticmethod
    #@tf.contrib.eager.defun
    def shift_left(tensor, label_tensor, numpy_svd=False, D=None, trunc_thresh=None):
        t = misc_mps.ncon([tensor, label_tensor],[[-1,1,-4], [1,-2,-3]])
        return MPSClassifier.split_off(t, direction='l', numpy_svd=numpy_svd, D=D, trunc_thresh=trunc_thresh)    

    
    # @staticmethod
    # #@tf.contrib.eager.defun    
    # def shift_right(label_tensor, tensor, numpy_svd=False, D=None, trunc_thresh=None):
    #     Nl = label_tensor.get_shape()[1]
    #     Nr = tensor.get_shape()[2]
    #     swap = np.zeros((Nl, Nr, Nr, Nl), dtype=tensor.dtype.as_numpy_dtype)
    #     for i in range(Nl):
    #         for j in range(Nr):
    #             swap[i,j,j,i] = 1

    #     net = tn.TensorNetwork()
    #     l_node =  net.add_node(label_tensor)
    #     t_node = net.add_node(tensor)
    #     swap_node = net.add_node(swap)
    #     #print('label.shape: ',label_tensor.shape, 'tensor.shpae: ', tensor.shape)
    #     e1 = net.connect(l_node[2], t_node[0])
    #     e2 = net.connect(l_node[1], swap_node[0])
    #     e3 = net.connect(t_node[2], swap_node[1])        
    #     left_edges = [l_node[0],  swap_node[2]]
    #     right_edges = [swap_node[3], t_node[1]]        
    #     temp = net.contract(e1)
    #     res = net.contract_between(temp, swap_node)
    #     if not numpy_svd:
    #         u_node, s_node, v_node, _ = net.split_node_full_svd(res, left_edges, right_edges, max_singular_values=D, max_truncation_err=trunc_thresh)
    #         Z = tf.linalg.norm(s_node.tensor)
    #         s_node.tensor /= Z
    #         out = u_node.reorder_axes([0, 2, 1])
    #         label_tensor = net.contract(s_node[1])
    #         return out.tensor , label_tensor.tensor
    #     else:
    #         out, label_tensor = split_node_full_svd_numpy(res, left_edges, right_edges, direction='r', max_singular_values=D, trunc_thresh=trunc_thresh)
    #         #print('out.shape: ',out.shape, 'label.shape: ', label_tensor.shape)            
    #         return out, label_tensor

    
    # @staticmethod
    # #@tf.contrib.eager.defun
    # def shift_left(tensor, label_tensor, numpy_svd=False, D=None, trunc_thresh=None):
    #     Nl = tensor.get_shape()[2]
    #     Nr = label_tensor.get_shape()[1]
    #     swap = np.zeros((Nl, Nr, Nr, Nl), dtype=tensor.dtype.as_numpy_dtype)
    #     for i in range(Nl):
    #         for j in range(Nr):
    #             swap[i,j,j,i] = 1
    #     net = tn.TensorNetwork()
    #     l_node =  net.add_node(label_tensor)
    #     t_node = net.add_node(tensor)
    #     swap_node = net.add_node(swap)
    #     e1 = net.connect(t_node[1],l_node[0])
    #     e2 = net.connect(t_node[2], swap_node[0])
    #     e3 = net.connect(l_node[1], swap_node[1])        
    #     left_edges = [t_node[0],  swap_node[2]]
    #     right_edges = [l_node[2], swap_node[3]]
    #     temp = net.contract(e1)
    #     res = net.contract_between(temp, swap_node)
    #     if not numpy_svd:
    #         u_node, s_node, v_node, _ = net.split_node_full_svd(res, left_edges, right_edges, max_singular_values=D, max_truncation_err=trunc_thresh)
    #         Z = tf.linalg.norm(s_node.tensor)
    #         s_node.tensor /= Z
    #         label_tensor = net.contract(s_node[0])
    #         return label_tensor.tensor, v_node.tensor
    #     else:
    #         #the numpy truncation scheme is different from tensorflow above
    #         label_tensor, out = split_node_full_svd_numpy(res, left_edges, right_edges, direction='l', max_singular_values=D,
    #                                                       trunc_thresh=trunc_thresh)
    #         return label_tensor, out



    @property
    def pos(self):
        return self._position
    @property
    def Dl(self):
        out = {n:self.tensors[n].shape[0] for n in range(self.pos)}
        out[self.pos] = self.tensors[self.pos-1].shape[1]
        return out
    @property
    def Dr(self):
        out = {n: self.tensors[n].shape[0] for n in range(self.pos,len(self))}
        out[len(self)] = self.tensors[-1].shape[1]
        return out
                                                                                 
    @property
    def D(self):
        raise NotImplementedError()
        
    def position(self, bond, numpy_svd=False, D=None, trunc_thresh=None):
        """
        """
        if bond == self.pos:
            return
        if bond > self.pos:
            for n in range(self._position, min(bond,len(self))):
                self.tensors[n],  self.label_tensor = self.shift_right(self.label_tensor, self.tensors[n], numpy_svd=numpy_svd, D=D, trunc_thresh=trunc_thresh)
                #print(n, self.tensors[n].shape,  self.label_tensor.shape)
            self._position = min(bond,len(self))
        if bond < self._position:
            for n in range(self._position - 1, max(-1,bond - 1), -1):
                #print(" shifting label tensor from site {0} to  site {1}".format(n, n-1))
                #print("shape(tensor[{0}]): {1}, shape(label_tensor): {2}", n, self.tensors[n].shape,  self.label_tensor.shape)                                
                self.label_tensor, self.tensors[n] = self.shift_left(self.tensors[n], self.label_tensor, numpy_svd=numpy_svd, D=D, trunc_thresh=trunc_thresh)
                #print("shape(tensor[{0}]): {1}, shape(label_tensor): {2}", n, self.tensors[n].shape,  self.label_tensor.shape)                                                

            self._position = max(0,bond)

        # for s in reversed(range(self.pos)):
        #     if s in self.right_data_environment:
        #         del self.right_data_environment[s]
        # for s in range(self.pos + 1, len(self) + 1):
        #     if s in self.left_data_environment:            
        #         del self.left_data_environment[s]
            
    def get_central_one_site_tensor(self, which): 
        """
        return the label_tensor contracted with the right-next mps tensor
        order of returned tensor: (Dl, n_labels, Dr, dl)
           1
           |
        0- O - O -2
               |
               3
        """
        if which in ('r','right'):
            return misc_mps.ncon([self.label_tensor, self.tensors[self.pos]],[[-1, -2, 1], [1, -3, -4]])
        elif which in ('l','left'):
            return misc_mps.ncon([self.tensors[self.pos - 1], self.label_tensor],[[-1, 1, -4], [1, -2, -3]])
    
    def get_central_two_site_tensor(self): 
        """
        return the label_tensor contracted with the right-next mps tensor
        order of returned tensor: (Dl, n_labels, Dr, dl_left, dl_right)
               1    
               |   
        0- O - O - O - 2
           |       |
           3       4
        """
        return misc_mps.ncon([self.tensors[self.pos - 1], 
                          self.label_tensor, self.tensors[self.pos]],
                         [[-1, 1, -4], [1, -2, 2], [2, -3, -5]])   
    
    def add_layer(self, embedded_data, site, direction):
        """
        add the data at site `site` to the environments
        Args:
            embedded_data (np.ndarray): shape (Nt,dl,N); the data matrix
            site (int):   the site of the system (i.e. the feature number)
            direction (int or str)
        """
        if direction in (1,'l','left'):
            assert(self.pos>site)
            if site == 0:
                self.left_data_environment[1] = misc_mps.ncon([self.tensors[0],embedded_data[:,:,0]],
                                                     [[-2,-3,1],[-1,1]]) #has shape (Nt, 1, D[1])  
            else:
                tensor = misc_mps.ncon([self.tensors[site],embedded_data[:,:,site]],
                               [[-2,-3,1],[-1,1]]) #has shape (Nt, D[site], D[site+1])
                #use tf.matmul with broadcasting to multiply the right-next vectors
                #has shape (Nt, 1, D[site + 1])
                self.left_data_environment[site + 1] = tf.matmul(self.left_data_environment[site], tensor) 
            out = tf.linalg.norm(self.left_data_environment[site + 1], axis=2) #get the norm of each row
            self.left_data_environment[site + 1] = tf.expand_dims(tf.squeeze(self.left_data_environment[site + 1], 1)/out,1)

                
        if direction in (-1,'r','right'):
            assert(self.pos <= site)
            if site == (len(self) - 1):

                self.right_data_environment[site - 1] = misc_mps.ncon([self.tensors[site],embedded_data[:,:,site]],
                                         [[-2,-3,1],[-1,1]]) #has shape (Nt, D[-1], 1)
                #print('shape of right_envs[{0}]'.format(site-1),self.right_data_environment[site - 1].shape)
                
            else:
                #first contract the image pixels into the mps tensor
                #print('shape of tensors[{0}]'.format(site),self.tensors[site].shape)
                tensor = misc_mps.ncon([self.tensors[site],embedded_data[:,:,site]],
                                   [[-2,-3,1],[-1, 1]]) #has shape (Nt, D[site], D[site+1])
                #use tf.matmul with broadcasting to multiply the right-next vectors
                #has shape (Nt, D[site],1)
                #print('shape of contracted tensor: ', tensor.shape)
                #print('shape of right_env[{0}]'.format(site), self.right_data_environment[site].shape)
                self.right_data_environment[site - 1] = tf.matmul(tensor, self.right_data_environment[site])
                #print('shape of right_envs[{0}]'.format(site-1),self.right_data_environment[site - 1].shape)                
            out = tf.linalg.norm(self.right_data_environment[site - 1], axis=1) #get the norm of each row
            #print('out.shape ', out.shape)
            self.right_data_environment[site - 1]= tf.expand_dims(tf.squeeze(self.right_data_environment[site - 1], 2)/out,2)
            
    def compute_data_environments(self,embedded_data):
        for site in range(self.pos):
            self.add_layer(embedded_data, site, 'l')
        for site in reversed(range(self.pos,len(self))):
            self.add_layer(embedded_data, site, 'r')
            
    def predict(self, embedded_data, which, debug=False):
        """
        Args:
            label_tensor (np.ndarray):  rank-4 array of shape (Dl, n_labels, Dr, dl)
            left_envs (np.ndarrray):    left data environments of shape (Nt, Dl)
            irght_envs: (np.ndarray):   right data environments of shape (Nt, Dr)
            embedded_data (np.ndarray): shape (Nt,dl,N); the data matrix
        Returns:
            np.ndarray of shape (Nt, n_labels)
        """
        if which in ('r', 'right'):
            assert(self.pos < len(self))
            
            #TODO: it might be better to do the contraction broadcastet; check this!
            label_tensor = self.get_central_one_site_tensor(which)
            #label_tensor = misc_mps.ncon([self.label_tensor, self.tensors[self.pos]],[[-1, -2, 1], [1, -3, -4]])
            Nt = embedded_data.shape[0]
            Dl, n_labels, Dr, dl = label_tensor.shape
            if self.pos > 0:
                left_envs = tf.squeeze(self.left_data_environment[self.pos], 1) #dummy index can be dropped 
                                                                             #because we're using tensordot instead of 
                                                                             #matmul
            else:
                left_envs = tf.ones(shape = (Nt,1), dtype=self.dtype) #FIXME: that's overkill; a single vector is enough
            if self.pos < len(self) - 1:
                right_envs = self.right_data_environment[self.pos]
            else:
                right_envs = tf.ones(shape=(Nt, 1, 1), dtype=self.dtype) #need a third dummy index for matmul
            bottom = tf.expand_dims(embedded_data[:,:,self.pos], 2)


 
        if which in ('l', 'left'):
            assert(self.pos > 0)
            
            #TODO: it might be better to do the contraction broadcastet; check this!
            label_tensor = self.get_central_one_site_tensor(which)
            #label_tensor = misc_mps.ncon([self.label_tensor, self.tensors[self.pos]],[[-1, -2, 1], [1, -3, -4]])
            Nt = embedded_data.shape[0]
            Dl, n_labels, Dr, dl = label_tensor.shape
            if self.pos > 1:
                left_envs = tf.squeeze(self.left_data_environment[self.pos - 1], 1) #dummy index can be dropped 
                                                                                 #because we're using tensordot instead of 
                                                                                 #matmul
            else:
                left_envs = tf.ones(shape=(Nt,1), dtype=self.dtype) #FIXME: that's overkill; a single vector is enough
                
            if self.pos < len(self):
                right_envs = self.right_data_environment[self.pos - 1]
            else:
                right_envs = tf.ones(shape=(Nt, 1, 1), dtype=self.dtype) #need a third dummy index for matmul
            bottom = tf.expand_dims(embedded_data[:,:,self.pos - 1],2)


        #contract left normally
        if debug:
            print('pos = {}, len(self) ={}'.format(self.pos, len(self)))
            print('left shape:',left_envs.shape)
            print('label shape:', label_tensor.shape)
            print('right shape:',right_envs.shape)
        # print()
        # print('maximums and minimums left_envs')
        # print(tf.math.reduce_max(left_envs),tf.math.reduce_min(left_envs))
        
        # print('maximums and minimums right_envs')
        # print(tf.math.reduce_max(right_envs),tf.math.reduce_min(right_envs))

        # print('maximums and minimums label_tensor')
        # print(tf.math.reduce_max(label_tensor),tf.math.reduce_min(label_tensor))

            
        t1 = tf.tensordot(left_envs, label_tensor, ([1],[0]))
        if debug:
            print('shape of t1 (left * label_tensor): {}'.format(t1.shape))
        #transpose 
        t2 = tf.transpose(t1,(0,1,3,2))
        if debug:
            print('shape of t2 (transposed t1) {}'.format(t2.shape))

        t3 = tf.reshape(t2,(Nt, n_labels*dl,Dr))
        if debug:
            print('shape of t3 (reshaped t2) {}'.format(t3.shape))

        #now contract the right using broadcasted matmul
        if debug:
            print('shape of right {}'.format(right_envs.shape))        
        t4 = tf.matmul(t3,right_envs)
        if debug:
            print('shape of t4 (t3*right) {}'.format(t4.shape))

        t5 = tf.reshape(t4,(Nt, n_labels, dl))
        if debug:
            print('shape of t5 (reshaped t4) {}'.format(t5.shape))
        t6 = tf.squeeze(tf.matmul(t5,bottom))
        if debug:
            print('shape of bottom:', bottom.shape)
            print('shape of t6 (t5 * bottom) ', t6.shape)
        norms = tf.expand_dims(tf.linalg.norm(t6, axis=1),1)
        t7 = t6 / norms #normalize the predictions
        return t7, norms  
        
    def one_site_gradient(self, embedded_data, labels, which): 
        """
        Args:
            embedded_data (np.ndarray):  shape (Nt, d, N) with Nt number of samples
                                         d the embedding dimension, and N the number of features
            labels (np.ndarray):         shape (Nt, n_labels): one-hot encoded labels for the 
                                         data in `embedded_data`
        Returns:
            np.ndarray of shape (Dl, n_labels, Dr, d)
        """
        if which in ('r','right'):
            n_labels = labels.shape[1]
            Nt = labels.shape[0]
            if self.pos > 0:
                left_env = tf.squeeze(self.left_data_environment[self.pos], 1)
                Dl = left_env.shape[1]
            else:
                left_env = tf.ones(shape=(Nt,1),  dtype=self.dtype) #FIXME: that's overkill; a single vector is enough        
                Dl = 1
            if self.pos < len(self) - 1:
                right_env = tf.squeeze(self.right_data_environment[self.pos], 2)
                Dr = right_env.shape[1]
            else:
                right_env = tf.ones(shape=(Nt, 1), dtype=self.dtype)
                Dr = 1
            dl = embedded_data.shape[1]
            # print()
            # print('gradient-predict')
            predict, norms = self.predict(embedded_data, which) #predictions are already normalized 
            #print(predict.dot(predict.T))
            #predict.shape is (Nt,n_labels)
            #each row in `predict` gets multiplied by `temp`.
            #`temp` is a set of `Nt` numbers, obtained from contracting the label vector for sample `n`
            #with the prediction vector for sample `n`.
            temp = tf.squeeze(tf.matmul(np.expand_dims(predict,1), np.expand_dims(labels, 2)), 1)      
            y = (predict * temp - labels)/norms
            loss = 1/2 * tf.math.reduce_sum((predict - labels)**2)
            t = batched_kronecker(batched_kronecker(batched_kronecker(left_env, 
                                                                      embedded_data[:, :, self.pos]),
                                                    right_env), 
                                  y)
            gradient = tf.math.reduce_sum(t,axis=0)

            return tf.transpose(tf.reshape(gradient, (Dl, dl, Dr, n_labels)),(0,3,2,1)), loss
        
        if which in ('l','left'):    
            n_labels = labels.shape[1]
            Nt = labels.shape[0]
            if self.pos  > 1:
                left_env = tf.squeeze(self.left_data_environment[self.pos - 1], 1)
                Dl = left_env.shape[1]
            else:
                left_env = tf.ones(shape=(Nt,1), dtype=self.dtype) #FIXME: that's overkill; a single vector is enough        
                Dl = 1
            if self.pos < len(self):
                right_env = tf.squeeze(self.right_data_environment[self.pos - 1], 2)
                Dr = right_env.shape[1]
            else:
                right_env = tf.ones(shape=(Nt, 1), dtype=self.dtype)
                Dr = 1
            dl = embedded_data.shape[1]
            # print('gradient-predict')
            predict, norms = self.predict(embedded_data,which) #predictions are already normalized 
            temp = tf.squeeze(tf.matmul(tf.expand_dims(predict,1), tf.expand_dims(labels, 2)), 1)
            y = (predict * temp - labels)/norms

            loss = 1/2 * tf.math.reduce_sum((predict - labels)**2)
            t = batched_kronecker(batched_kronecker(batched_kronecker(left_env, 
                                                                      embedded_data[:,:,self.pos  - 1]),
                                                    right_env), 
                                  y)
            gradient = tf.math.reduce_sum(t,axis=0)

            return tf.transpose(tf.reshape(gradient, (Dl, dl, Dr, n_labels)), (0,3,2,1)), loss            
    
    
    def do_one_site_step(self, embedded_data, labels, direction, learning_rate=1E-5, numpy_svd=False, loss_thresh=1.0,
                         max_singular_values=None, trunc_thresh=None):
        """
        learning rate should be positive
        """
        if direction in ('right','r'):
            old_tensor = copy.copy(self.tensors[self.pos])
            old_label_tensor = copy.copy(self.label_tensor)
            
            gradient, loss = self.one_site_gradient(embedded_data, labels, which='r')
            gradient /= tf.linalg.norm(gradient)
            #merge the label-tensor into the mps from the left
            temp = self.get_central_one_site_tensor(which = 'r')
            temp = (temp - learning_rate * gradient)
            self.tensors[self.pos],  self.label_tensor = self.split_off(temp, direction='r', numpy_svd=numpy_svd, D=max_singular_values, trunc_thresh=trunc_thresh)                

            # net = tn.TensorNetwork()
            # node = net.add_node(temp)
            # left_edges = [node[0],  node[3]]
            # right_edges = [node[1], node[2]]
            # if not numpy_svd:
            #     u_node, s_node, v_node, _ = net.split_node_full_svd(node, left_edges, right_edges, max_singular_values=max_singular_values, max_truncation_err=trunc_thresh)
            #     Z = tf.linalg.norm(s_node.tensor)
            #     s_node.tensor /= Z
            #     out = u_node.reorder_axes([0, 2, 1])
            #     label_tensor = net.contract(s_node[1])
                
            #     self.tensors[self.pos] = out.tensor
            #     self.label_tensor = label_tensor.tensor
            # else:
            #     out, label_tensor = split_node_full_svd_numpy(node, left_edges, right_edges, direction='r', max_singular_values=max_singular_values, max_truncation_err=trunc_thresh)
            #     self.tensors[self.pos] = out
            #     self.label_tensor = label_tensor
                
            self._position += 1
            self.add_layer(embedded_data, self.pos - 1, direction=1)
            
            predict, norms = self.predict(embedded_data, which='r')
            new_loss = 1/2 * tf.math.reduce_sum((predict - labels)**2)
            if (new_loss - loss) > loss_thresh:
                self._position -= 1
                self.tensors[self.pos] = old_tensor
                self.label_tensor = old_label_tensor
                self.add_layer(embedded_data, self.pos, direction=-1)                                
            
            
        if direction in ('l','left'):
            old_tensor = copy.copy(self.tensors[self.pos - 1])
            old_label_tensor = copy.copy(self.label_tensor)
            
            gradient, loss = self.one_site_gradient(embedded_data, labels, which='l')
            gradient /= np.linalg.norm(gradient)
            #merge the label-tensor into the mps from the right
            temp = self.get_central_one_site_tensor(which = 'l')
            temp = (temp - learning_rate * gradient)

            self.label_tensor, self.tensors[self.pos - 1] = self.split_off(temp, direction='l', numpy_svd=numpy_svd, D=max_singular_values, trunc_thresh=trunc_thresh)                            
            # net = tn.TensorNetwork()
            # node = net.add_node(temp)
            # left_edges = [node[0],  node[1]]
            # right_edges = [node[2], node[3]]
            # if not numpy_svd:
            #     u_node, s_node, v_node, _ = net.split_node_full_svd(node, left_edges, right_edges, max_singular_values=max_singular_values, max_truncation_err=trunc_thresh)
            #     Z = tf.linalg.norm(s_node.tensor)
            #     s_node.tensor /= Z
            #     label_tensor = net.contract(s_node[0])

            #     self.tensors[self.pos - 1] = v_node.tensor
            #     self.label_tensor = label_tensor.tensor
            # else:
            #     label_tensor, out = split_node_full_svd_numpy(node, left_edges, right_edges, direction='l', max_singular_values=max_singular_values, max_truncation_err=trunc_thresh)
            #     self.tensors[self.pos - 1] = out
            #     self.label_tensor = label_tensor
            
            self._position -= 1
            self.add_layer(embedded_data, self.pos, direction=-1)
            predict, norms = self.predict(embedded_data, which='r')
            new_loss = 1/2 * tf.math.reduce_sum((predict - labels)**2)
            if (new_loss - loss) > loss_thresh:
                self._position += 1
                self.tensors[self.pos - 1] = old_tensor
                self.label_tensor = old_label_tensor
                self.add_layer(embedded_data, self.pos - 1, direction=1)                
            
        return loss, gradient


    
    def right_sweep_simple(self, samples, labels, learning_rate, numpy_svd=False,
                           D=None,
                           trunc_thresh=1E-8):
        losses = []
        train_accuracies = []

        ground_truth = tf.argmax(labels,  axis=1)
        lr = learning_rate
        while self.pos < len(self) - 1:
            loss, gradient = self.do_one_site_step(samples, 
                                                   labels, learning_rate=lr, 
                                                   direction='r', numpy_svd=numpy_svd,
                                                   loss_thresh=1000000000000.0,
                                                   max_singular_values=D,
                                                   trunc_thresh=trunc_thresh)
            if self.pos < len(self):
                prediction = tf.argmax(self.predict(samples, which='r', debug=False)[0], 1)
            else:
                prediction = tf.argmax(self.predict(samples, which='l', debug=False)[0], 1)
            correct = np.sum(prediction.numpy()==ground_truth.numpy())
            train_accuracies.append(correct/labels.shape[0])
            losses.append(loss)
            stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f, D=%i" % (self.pos, 
                                                                                                 np.real(loss), 
                                                                                                 np.imag(loss), 
                                                                                                 lr, 
                                                                                                 correct/labels.shape[0], self.tensors[self.pos - 1].shape[1]))
            stdout.flush()
        with open(self.name + 'data_.pickle', 'wb') as f:
            pickle.dump({'losses':losses, 'accuracies': train_accuracies},f)
        return losses, train_accuracies            
                
    def left_sweep_simple(self, samples, labels, learning_rate, numpy_svd=False,
                          D=None,
                          trunc_thresh=1E-8):
                          
        losses = []
        train_accuracies = []

        ground_truth = tf.argmax(labels,  axis=1)
        lr = learning_rate
                        
        while self.pos < len(self):
            self.position(self.pos + 1, trunc_thresh=trunc_thresh)

            self.add_layer(samples, self.pos - 1, direction=1)                    
        while self.pos > 1:
            loss, gradient = self.do_one_site_step(samples, 
                                                   labels, learning_rate=lr, 
                                                   direction='l', numpy_svd=numpy_svd,
                                                   loss_thresh=1000000000000.0,
                                                   max_singular_values=D,                                                       
                                                   trunc_thresh=trunc_thresh)
            if self.pos < len(self):
                prediction = tf.argmax(self.predict(samples, which='r', debug=False)[0], 1)
            else:
                prediction = tf.argmax(self.predict(samples, which='l', debug=False)[0], 1)
                
            correct = np.sum(prediction.numpy()==ground_truth.numpy())
            train_accuracies.append(correct/labels.shape[0])                
            losses.append(loss)
            stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f, D=%i" % (self.pos, 
                                                                                                 np.real(loss), 
                                                                                                 np.imag(loss), 
                                                                                                 lr, 
                                                                                                 correct/labels.shape[0], self.tensors[self.pos].shape[0]))
            stdout.flush()
        with open(self.name + 'data_.pickle', 'wb') as f:
            pickle.dump({'losses':losses, 'accuracies': train_accuracies},f)
                
        return losses, train_accuracies


    def optimize_simple(self, samples, labels, learning_rate, num_sweeps, numpy_svd=False,
                        D=None,
                        trunc_thresh=1E-8):
        losses, accuracies = [],[]        
        for sweep in range(num_sweeps):
            loss, accs = self.right_sweep_simple(samples, labels, learning_rate, numpy_svd=False,
                                                 D=None,
                                                 trunc_thresh=1E-8)
            losses.extend(loss)
            accuracies.extend(accs)
            
            loss, accs = self.left_sweep_simple(samples, labels, learning_rate, numpy_svd=False,
                                                D=None,
                                                trunc_thresh=1E-8)
            
            losses.extend(loss)
            accuracies.extend(accs)
        return losses, accuracies
    
    def optimize(self, samples, labels, learning_rate, num_sweeps, n0=0, numpy_svd=False,
                 factor=1.5, n_stpcnt=10,
                 lower_bound=1E-10,
                 max_local_steps=4, loss_thresh=1.0,
                 D=None,
                 trunc_thresh=1E-8):
        losses = []
        train_accuracies = []
        walltimes = []

        self.position(0, numpy_svd=numpy_svd, trunc_thresh=trunc_thresh)        
        self.position(n0, numpy_svd=numpy_svd, trunc_thresh=trunc_thresh)
        self.compute_data_environments(samples)
        ground_truth = tf.argmax(labels,  axis=1)
        old_loss = 1E200
        t1 = time.time()
        lr = learning_rate
        stpcnt = 0
        self.position(n0, numpy_svd=numpy_svd, trunc_thresh=trunc_thresh)
        self.compute_data_environments(samples)
        cnt_steps = 0
        for sweep in range(num_sweeps):
            # if sweep < len(learning_rates):
            #     lr = learning_rates[sweep]
            # else:
            #     lr = learning_rates[-1]
            while self.pos > n0:
                self.position(self.pos - 1, trunc_thresh=trunc_thresh)
                self.add_layer(samples, self.pos, direction=-1)
            
            while self.pos < len(self) - n0 - 1:
                #for site in range(n0, len(self) - n0):
                old_pos = self.pos
                if cnt_steps < max_local_steps:
                    loss, gradient = self.do_one_site_step(samples, 
                                                           labels, learning_rate=lr, 
                                                           direction='r', numpy_svd=numpy_svd,
                                                           max_singular_values=D,                                                           
                                                           loss_thresh=loss_thresh)
                else:
                    self.position(self.pos + 1, trunc_thresh=trunc_thresh)
                    self.add_layer(samples, self.pos - 1, direction=1)                    
                    cnt_steps = 0
                    
                if self.pos == old_pos:
                    cnt_steps += 1
                    
                if self.pos < len(self):
                    prediction = tf.argmax(self.predict(samples, which='r', debug=False)[0], 1)
                else:
                    prediction = tf.argmax(self.predict(samples, which='l', debug=False)[0], 1)
                correct = np.sum(prediction.numpy()==ground_truth.numpy())
                train_accuracies.append(correct)
                if ((loss - old_loss) > loss_thresh) and (np.abs(lr) > np.abs(lower_bound)):                
                    lr /= factor
                    stpcnt = 0
                elif ((loss - old_loss) <= loss_thresh) and (stpcnt > n_stpcnt) and (lr < learning_rate):
                    lr *= factor
                elif ((loss - old_loss) <= loss_thresh) and (stpcnt > n_stpcnt) and (lr >= learning_rate):                    
                    lr = learning_rate

                stpcnt += 1
                old_loss = loss
                losses.append(loss)
                stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f" % (self.pos, 
                                                                                               np.real(loss), 
                                                                                               np.imag(loss), 
                                                                                               lr, 
                                                                                               correct/labels.shape[0]))
                stdout.flush()
            walltimes.append(time.time() - t1)
            t1 = time.time()
            cnt_steps = 0
                     
            while self.pos < len(self) - n0:
                self.position(self.pos + 1, trunc_thresh=trunc_thresh)
                self.add_layer(samples, self.pos - 1, direction=1)                    
            while self.pos > n0 + 1:
                old_pos = self.pos
                if cnt_steps < max_local_steps:
                    loss, gradient = self.do_one_site_step(samples, 
                                                           labels, learning_rate=lr, 
                                                           direction='l', numpy_svd=numpy_svd,
                                                           max_singular_values=D,                                                           
                                                           loss_thresh=loss_thresh)
                else:
                    self.add_layer(samples, self.pos, direction=-1)                                                            
                    self.position(self.pos - 1,  trunc_thresh=trunc_thresh)
                    self.add_layer(samples, self.pos, direction=-1)                                        
                    cnt_steps = 0
                    
                if self.pos == old_pos:
                    cnt_steps += 1
                                
                if self.pos < len(self):
                    prediction = tf.argmax(self.predict(samples, which='r', debug=False)[0], 1)
                else:
                    prediction = tf.argmax(self.predict(samples, which='l', debug=False)[0], 1)
                    
                correct = np.sum(prediction.numpy()==ground_truth.numpy())
                train_accuracies.append(correct)
                if ((loss - old_loss) > loss_thresh) and (np.abs(lr) > np.abs(lower_bound)): 
                    lr /= factor
                    stpcnt = 0
                elif ((loss - old_loss) <= loss_thresh) and (stpcnt > n_stpcnt) and (lr < learning_rate):
                    lr *= factor
                elif ((loss - old_loss) <= loss_thresh) and (stpcnt > n_stpcnt) and (lr >= learning_rate):                    
                    lr = learning_rate

                stpcnt += 1
                old_loss = loss
                
                losses.append(loss)                
                stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f" % (self.pos, 
                                                                                               np.real(loss), 
                                                                                               np.imag(loss), 
                                                                                               lr, 
                                                                                               correct/labels.shape[0]))
                
                stdout.flush()
            walltimes.append(time.time() - t1)
            with open(self.name + '.pickle', 'wb') as f:
                pickle.dump({'walltimes' : walltimes, 'losses':losses, 'accuracies': train_accuracies},f)
                
        return losses, train_accuracies, walltimes
    
def load_MNIST(folder):
    """
    load the MNIST data
    Returns:
        train (list): list of length two containing the flattened images and the corresponding labels
        valid (list): list of length two containing the flattened images and the corresponding labels  
        tset (list):  list of length two containing the flattened images and the corresponding labels
    """
    print('loading MNIST data')
    train_set=[np.load(folder+'/train_set_data.npy'),np.load(folder+'/train_set_label.npy')]
    valid_set=[np.load(folder+'/valid_set_data.npy'),np.load(folder+'/valid_set_label.npy')]
    test_set=[np.load(folder+'/test_set_data.npy'),np.load(folder+'/test_set_label.npy')]
    return train_set,valid_set,test_set

    
def generate_mapped_MNIST_batches(X,Y,n_batches):
    """
    X is an M by N matrix, where M is the number of samples and N is the number of features 
    (N= 28*28 for the MNIST data set) Y are the labels 
    returns [f_1(x),f_2(x),Y], where f_1=cos(pi*x*256/(2*255)) and f_2=sin(pi*x*256/(2*255))
    pass this to calculateGradient
    """
    batch_size = X.shape[0] // n_batches
    nb = len(np.unique(Y))
    y_one_hot = [np.eye(nb)[np.array([Y[n*batch_size:(n+1)*batch_size]])].squeeze().astype(np.int32)  
                 for n in range(n_batches)]
    X_mapped = [np.transpose(np.array([np.cos((X[n*batch_size:(n+1)*batch_size,:]*256)/255*math.pi/2),
                          np.sin((X[n*batch_size:(n+1)*batch_size,:]*256)/255*math.pi/2)]),(1,0,2)) 
                for n in range(n_batches)]    
    return X_mapped, y_one_hot

def generate_mapped_MNIST_batches_poly(data,labels,n_batches):
    """
    X is an M by N matrix, where M is the number of samples and N is the number of features 
    (N= 28*28 for the MNIST data set) Y are the labels 
    returns [f_1(x),f_2(x),Y], where f_1=cos(pi*x*256/(2*255)) and f_2=sin(pi*x*256/(2*255))
    pass this to calculateGradientn
    """
    X = data / 255
    Y = labels
    batch_size = X.shape[0] // n_batches
    nb = len(np.unique(Y))
    y_one_hot = [np.eye(nb)[np.array([Y[n*batch_size:(n+1)*batch_size]])].squeeze().astype(np.int32)  
                 for n in range(n_batches)]
    X_mapped = [np.transpose(np.array([1 - X[n*batch_size:(n+1)*batch_size,:],
                                       X[n*batch_size:(n+1)*batch_size,:]]),(1,0,2)) 
                for n in range(n_batches)]    
    return X_mapped, y_one_hot    
def generate_MNIST_mapped(X,Y):
    """
    X is an M by N matrix, where M is the number of samples and N is the number of features 
    (N= 28*28 for the MNIST data set) Y are the labels 
    returns [f_1(x),f_2(x),Y], where f_1=cos(pi*x*256/(2*255)) and f_2=sin(pi*x*256/(2*255))
    pass this to calculateGradient
    """

    nb = len(np.unique(Y))
    y_one_hot = np.eye(nb)[np.array([Y])].squeeze().astype(np.int32)  
                 
    embedded_data = [np.array([np.cos((X[n,:]*256)/255*math.pi/2),np.sin((X[n,:]*256)/255*math.pi/2)])
                    for n in range(X.shape[0])]    
    return embedded_data,y_one_hot

 
def batched_kronecker(a,b):   
    """
    compute the kronecker product np.kron(a[n,:], b[n,:]) for all `n` using 
    numpy broadcasting
    Args:
        a (np.ndarray):  matrix of shape (Nt, d)
        b (np.ndarray):  matrix of shape (Nt, d)
    Returns:
        np.ndarray of shape (Nt, d**2)
    """
    return tf.reshape(tf.matmul(tf.expand_dims(a,2),tf.expand_dims(b,1)),(a.shape[0],a.shape[1]*b.shape[1]))





    
