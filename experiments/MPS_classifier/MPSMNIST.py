import sys
from sys import stdout
import tensorflow as tf
import copy
import numpy as np
import itertools
import math
import experiments.MPS.misc_mps as misc_mps
import experiments.MPS.matrixproductstates as mps
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


    
class MPSClassifier(mps.FiniteMPSCentralGauge):
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
    def eye(cls, ds, D, num_labels, label_position, dtype=tf.float64, noise=1E-5,scaling=0.1, name='MPS_classifier'):
        N = len(ds)
        Ds = [1] + [D] * N + [1]
        ds.insert(label_position, num_labels)
        tensors = [np.transpose(np.reshape(np.eye(Ds[n], Ds[n + 1])[list(range(Ds[n]))*ds[n]],(ds[n], 
                                                                                           Ds[n], Ds[n + 1])),
                               (1, 0, 2)).astype(dtype.as_numpy_dtype)
                   for n in range(N + 1)]
        for t in tensors:
            t += (np.random.random_sample(t.shape)).astype(dtype.as_numpy_dtype) * noise
            t *= scaling
        tf_tensors =  [tf.convert_to_tensor(t) for t in tensors]
        return cls(mps_tensors=tf_tensors, label_position=label_position, dtype=dtype, name=name)
    
    @classmethod
    def random(cls, ds, D, num_labels, label_position, dtype=tf.float64, scaling=0.1, name='MPS_classifier'):
        N = len(ds)
        Ds = [1] + [D] * N + [1]
        ds.insert(label_position, num_labels)
        tensors = [np.random.randn(Ds[n], ds[n], Ds[n + 1])*scaling
                   for n in range(N + 1)]
        tf_tensors =  [tf.convert_to_tensor(t) for t in tensors]
        return cls(mps_tensors=tf_tensors, label_position=label_position, dtype=dtype, name=name)    

    def __init__(self, mps_tensors, label_position, dtype, name = 'MPS_classifier'):
        self.right_data_environment={}
        self.left_data_environment={}
        self._label_position = label_position
        #self.dtype = dtype
        super().__init__(mps_tensors,tf.ones(shape=[1,1], dtype=dtype), 0,name)
        

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
                label_tensor = net.contract(s_node[1])
                return u_node.tensor , label_tensor.tensor
            else:
                out, label_tensor = split_node_full_svd_numpy(t_node, left_edges, right_edges, direction='r', max_singular_values=D, trunc_thresh=trunc_thresh)
                #print('out.shape: ',out.shape, 'label.shape: ', label_tensor.shape)            
                return tf.transpose(out,(0,2,1)), label_tensor

        if direction in ('l','left'):            
            net = tn.TensorNetwork()
            t_node = net.add_node(tensor)
            left_edges = [t_node[0],  t_node[1]]
            right_edges = [t_node[2], t_node[3]]
            if not numpy_svd:
                u_node, s_node, v_node, _ = net.split_node_full_svd(t_node, left_edges, right_edges, max_singular_values=D, max_truncation_err=trunc_thresh)
                out = v_node.reorder_axes([0, 2, 1])
                Z = tf.linalg.norm(s_node.tensor)
                s_node.tensor /= Z
                label_tensor = net.contract(s_node[0])
                return label_tensor.tensor, out.tensor
            else:
                #the numpy truncation scheme is different from tensorflow above
                label_tensor, out = split_node_full_svd_numpy(t_node, left_edges, right_edges, direction='l', max_singular_values=D,
                                                              trunc_thresh=trunc_thresh)
                return label_tensor, tf.transpose(out,(0,2,1))
                
    @staticmethod        
    def shift_right(label_tensor, tensor, numpy_svd=False, D=None, trunc_thresh=None):
        t = misc_mps.ncon([label_tensor, tensor],[[-1,-2,1], [1,-4,-3]])        
        return MPSClassifier.split_off(t, direction='r', numpy_svd=numpy_svd, D=D, trunc_thresh=trunc_thresh)    

    
    @staticmethod
    #@tf.contrib.eager.defun
    def shift_left(tensor, label_tensor, numpy_svd=False, D=None, trunc_thresh=None):
        t = misc_mps.ncon([tensor, label_tensor],[[-1, -4, 1], [1,-2,-3]])
        return MPSClassifier.split_off(t, direction='l', numpy_svd=numpy_svd, D=D, trunc_thresh=trunc_thresh)    

    @property
    def label_pos(self):
        return self._label_position

    
    def label_position(self, site, numpy_svd=False, D=None, trunc_thresh=None):
        if site == self._label_position:
            return
        elif site > self._label_position:
            for n in range(self._label_position, min(site,len(self))):
                self._tensors[n],  self._tensors[n+1] = self.shift_right(self.get_tensor(n), self.get_tensor(n + 1), numpy_svd=numpy_svd, D=D, trunc_thresh=trunc_thresh)
                #in case we have absorbed the centermatrix we need to reset it to 1
                #this case also covers self.pos = N
                if self.pos == (n+1):
                    self.mat = tf.eye(num_rows = self._tensors[n].shape[2], num_columns = self._tensors[n + 1].shape[0],
                                                dtype=self._tensors[n].dtype) 
                elif self.pos == n:                    
                    self.mat = tf.eye(num_rows =  self._tensors[n].shape[0],
                                      dtype=self._tensors[n].dtype) 
                    
            self._label_position = min(site,len(self))
        elif site < self._label_position:
            for n in reversed(range(site, self._label_position)):
                self._tensors[n], self._tensors[n + 1] = self.shift_left(self.get_tensor(n), self.get_tensor(n + 1), numpy_svd=numpy_svd, D=D, trunc_thresh=trunc_thresh)
                if self.pos == n:
                    self.mat = tf.eye(num_rows = self._tensors[n].shape[0],
                                                dtype=self._tensors[n].dtype) 
                elif self.pos == (n + 1):                    
                    self.mat = tf.eye(num_rows = self._tensors[n].shape[2],num_columns =  self._tensors[n + 1].shape[0],
                                                dtype=self._tensors[n].dtype) 
                
            self._label_position = max(0,site)

            
    def add_layer(self, samples, site, direction):
        """
        this computes the environments at `site + 1` or `site -1` for direction = 1 or -1, respectively,
        for the mps; site labels the mps sites, including the label-tensor. 
        this calls ._tensors, and not get_tensor, i.e. centermatrix is not absorbed into the mps
        left environments to the left of label_pos have dimensions (Nt, 1, D), to the right (Nt, Nlables, D)
        right environments to the right of label_pos have dimensions (Nt, D, 1), to the left (Nt, D, Nlables)
        Args:
            samples (np.ndarray): shape (Nt,dl,N); the data matrix
            site (int):   the site of the system (i.e. the feature number)
            direction (int or str)
        """
        
        if direction in (1,'l','left'):
            assert(self.pos > site)            
            if site < self._label_position:
                if site == -1:
                    self.left_data_environment[site + 1] =  tf.ones(shape = (samples.shape[0],1,1), dtype=self.dtype)
                    
                elif site == 0:
                    self.left_data_environment[site + 1] = misc_mps.ncon([self._tensors[0],samples[:,:,0]],
                                                                  [[-2, 1, -3],[-1, 1]]) #has shape (Nt, 1, D[1])  
                else:
                    tensor = misc_mps.ncon([self._tensors[site],samples[:,:,site]],
                                           [[-2, 1, -3],[-1, 1]]) #has shape (Nt, D[site], D[site+1])
                    #use tf.matmul with broadcasting to multiply the right-next vectors
                    #has shape (Nt, 1, D[site + 1])
                    self.left_data_environment[site + 1] = tf.matmul(self.left_data_environment[site], tensor) 
                norms = tf.linalg.norm(self.left_data_environment[site + 1], axis=2) #get the norm of each row
                self.left_data_environment[site + 1] = tf.expand_dims(tf.squeeze(self.left_data_environment[site + 1], 1)/norms,1)
            elif site == self._label_position:
                self.left_data_environment[site + 1] = misc_mps.ncon([tf.squeeze(self.left_data_environment[site],1),  self._tensors[site]],[[-1, 1],[1, -2, -3]])
                #has dimensions (Nt, Nlabels, D[site + 1])                
                Nt, n_labels, D = self.left_data_environment[site + 1].shape
                #normalize by the full tensor                                
                norms = tf.expand_dims(tf.expand_dims(tf.linalg.norm(tf.reshape(self.left_data_environment[site + 1], (Nt, n_labels * D)), axis=1), 1), 1) #(Nt, 1)
                self.left_data_environment[site + 1] = self.left_data_environment[site + 1] / norms

            elif site > self._label_position:

                tensor = misc_mps.ncon([self._tensors[site],samples[:, :, site - 1]],
                                       [[-2, 1, -3],[-1, 1]]) #has shape (Nt, D[site], D[site+1])
                #use tf.matmul with broadcasting to multiply the right-next vectors
                #self.left_data_environment[site] has shape (Nt, Nlabels, D[site])
                self.left_data_environment[site + 1] = tf.matmul(self.left_data_environment[site], tensor)  #has shape (Nt, Nlabels, D[site + 1])
                Nt, n_labels, D = self.left_data_environment[site + 1].shape
                #normalize by the full tensor                                
                norms = tf.expand_dims(tf.expand_dims(tf.linalg.norm(tf.reshape(self.left_data_environment[site + 1], (Nt, n_labels * D)), axis=1), 1), 1) #(Nt, 1)                
                self.left_data_environment[site + 1] = self.left_data_environment[site + 1]/norms
            return self.left_data_environment[site + 1]

        if direction in (-1,'r','right'):
            assert(self.pos <= site)
            if site > self._label_position:
                if site == len(self):#can only be here
                    self.right_data_environment[site - 1] =  tf.ones(shape = (samples.shape[0],1, 1), dtype=self.dtype)
                elif site == (len(self) - 1):
                    self.right_data_environment[site - 1] = misc_mps.ncon([self._tensors[site],samples[:, :, site - 1]],
                                                                          [[-2, 1, -3],[-1,1]]) #has shape (Nt, D[-1], 1)
                else:
                    #first contract the image pixels into the mps tensor
                    tensor = misc_mps.ncon([self._tensors[site],samples[:, :, site - 1]],
                                           [[-2, 1, -3],[-1, 1]]) #has shape (Nt, D[site], D[site+1])
                    #use tf.matmul with broadcasting to multiply the right-next vectors
                    #has shape (Nt, D[site],1)
                    self.right_data_environment[site - 1] = tf.matmul(tensor, self.right_data_environment[site])
                    
                norms = tf.linalg.norm(self.right_data_environment[site - 1], axis=1) #get the norm of each row
                #has shape (Nt, D[site], 1)
                self.right_data_environment[site - 1]= tf.expand_dims(tf.squeeze(self.right_data_environment[site - 1], 2)/norms,2)
            elif site == self._label_position:
                #has shape (Nt, D[site], Nlabels) 
                self.right_data_environment[site - 1] = tf.squeeze(misc_mps.ncon([self._tensors[site], self.right_data_environment[site]],[[-2, -3, 1],[-1, 1, -4]]), 3)
                Nt, D, n_labels = self.right_data_environment[site - 1].shape
                #normalize by the full tensor                
                norms = tf.expand_dims(tf.expand_dims(tf.linalg.norm(tf.reshape(self.right_data_environment[site - 1], (Nt, n_labels * D)), axis=1), 1), 1) #(Nt, 1)                                
                self.right_data_environment[site - 1] = self.right_data_environment[site - 1] / norms
                #print(site, tf.linalg.norm(self.right_data_environment[site - 1],axis=1))#checked that this is all ones
            elif site < self._label_position:
                #first contract the image pixels into the mps tensor
                tensor = misc_mps.ncon([self._tensors[site],samples[:,:,site]],
                                       [[-2, 1, -3],[-1, 1]]) #has shape (Nt, D[site], D[site+1])
                #use tf.matmul with broadcasting to multiply the right-next vectors
                #has shape (Nt, D[site], Nlabels)
                self.right_data_environment[site - 1] = tf.matmul(tensor, self.right_data_environment[site])
                Nt, D, n_labels = self.right_data_environment[site - 1].shape
                #normalize by the full tensor
                norms = tf.expand_dims(tf.expand_dims(tf.linalg.norm(tf.reshape(self.right_data_environment[site - 1], (Nt, n_labels * D)), axis=1), 1), 1) #(Nt, 1)                                
                self.right_data_environment[site - 1]= self.right_data_environment[site - 1]/norms
            return self.right_data_environment[site - 1]                

    def compute_data_environments(self,samples):
        for site in range(-1, self.pos):
            self.add_layer(samples, site, 'l')
        for site in reversed(range(self.pos,len(self) + 1)):
            self.add_layer(samples, site, 'r')
            
    def accuracy(self, samples, labels):
        ground_truth = tf.argmax(labels,  axis=1)        
        prediction = tf.argmax(self.predict(samples)[0], 1)
        correct = np.sum(prediction.numpy() == ground_truth.numpy())
        return correct/labels.shape[0]
    
    def predict(self, samples):
        """
        Args:
            label_tensor (np.ndarray):  rank-4 array of shape (Dl, n_labels, Dr, dl)
            left_envs (np.ndarrray):    left data environments of shape (Nt, Dl)
            irght_envs: (np.ndarray):   right data environments of shape (Nt, Dr)
            samples (np.ndarray): shape (Nt,dl,N); the data matrix
        Returns:
            predictions (tf.Tensor)     shape (Nt, n_labels))
            norms (tf.Tensor)           shape (Nt, 1)

        """
        if self.pos == self.label_pos:
            y = misc_mps.ncon([tf.squeeze(self.left_data_environment[self.label_pos], 1),
                                          self.get_tensor(self.label_pos)],
                                         [[-1, 1], [1, -2, -3]])
            predict = tf.squeeze(tf.matmul(y,self.right_data_environment[self.label_pos]),2)
            norms = tf.expand_dims(tf.linalg.norm(predict, axis=1), 1)
            predict = predict/norms
            return predict, norms
        elif self.pos > self.label_pos:
            assert(self.pos>0) #should be always the case
            #also need this for other parts
            left = self.left_data_environment[self.pos]                           #(Nt, n_labels, D)
            
            right = tf.squeeze(self.right_data_environment[self.pos], 2)             #(Nt, D)
            t0 = misc_mps.ncon([left, self.centermatrix],[[-1, -2, 1],[1, -3]])   #(Nt, n_labels, D)
            t2 = self.add_layer(samples, self.pos, -1)                            #(Nt, D, 1)
            t4 = tf.matmul(t0,t2)                                                 #(Nt, n_labels, 1)
            norms = tf.linalg.norm(t4, axis = 1)                                  #(Nt, 1)
            #the label tensor is on the left side of self.pos, index samples with self.pos-1\
            prediction = tf.squeeze(t4)/norms
            return prediction, norms
        
        elif self.pos < self.label_pos:
            left = self.left_data_environment[self.pos]                    #(Nt, 1, D)
            t0 = misc_mps.ncon([left, self.centermatrix],[[-1, -2, 1],[1, -3]])   #(Nt, 1, D)
            t1 = self.add_layer(samples, self.pos, -1)                            #(Nt, D, n_labels)
            t3 = tf.expand_dims(tf.squeeze(tf.matmul(t0, t1),1),2)                #(Nt, n_labels, 1)
            norms = tf.linalg.norm(t3, axis = 1)                                  #(Nt, 1)
            prediction = tf.squeeze(t3)/norms
            return prediction, norms
        
    def one_site_gradient(self, samples, labels): 
        """
        compute the gradient with respect to the tensor at site self.pos
        This routine assumes that the centermatrix is to the right of the tensor;
        thus it is not expected to give correct results for self.pos == len(self)
        Args:
            samples (tf.Tensor):         shape (Nt, d, N) with Nt number of samples
                                         d the embedding dimension, and N the number of features
            labels (tf.Tensor):          shape (Nt, n_labels): one-hot encoded labels for the 
                                         data in `samples`
        Returns:
            tf.Tensor of shape (Dl, n_labels, Dr, d)
        """
        #Todo:  this likely uses too much memory, fix this!
        assert(self.pos < len(self)) #exclude the case where central site is a the right boundary, see above
        if self.pos == self.label_pos:
            prediction, norms = self.predict(samples)
            Dr = self._tensors[self.pos].shape[2]
            Nt, n_labels = labels.shape            
            left = tf.squeeze(self.left_data_environment[self.pos], 1)
            right = tf.squeeze(self.right_data_environment[self.pos], 2)
            t1 = tf.squeeze(tf.matmul(tf.expand_dims(prediction,1),tf.expand_dims(labels,2)),1)

            y = (t1 * prediction - labels)/norms
            Dl = left.shape[1]
            grad = tf.math.reduce_mean(tf.reshape(batched_kronecker(batched_kronecker(left, y), right), (Nt, Dl, n_labels, Dr)), axis=0)
            loss = 1/2 * tf.math.reduce_mean(tf.math.reduce_sum((prediction - labels)**2, 1), 0)
            return grad, loss, prediction
        
        elif self.pos > self.label_pos:
            assert(self.pos>0) #should be always the case
            Dl = self._tensors[self.pos - 1].shape[2]
            d = self._tensors[self.pos].shape[1]
            Dr = self._tensors[self.pos].shape[2]
            
            Nt, n_labels = labels.shape
            left = self.left_data_environment[self.pos]                     #(Nt, n_labels, D)
            right = tf.squeeze(self.right_data_environment[self.pos], 2)       #(Nt, D)
            y = tf.squeeze(tf.matmul(tf.expand_dims(labels, 1),  left), 1)  #(Nt, D)
            
            #also need this for other parts 
            t0 = misc_mps.ncon([left, self.centermatrix],[[-1, -2, 1],[1, -3]])   #(Nt, n_labels, D)
            t1 = tf.matmul(tf.expand_dims(labels, 1), t0)                         #(Nt, 1, D)
            t2 = self.add_layer(samples, self.pos, -1)                            #(Nt, D, 1)
            factor = tf.squeeze(tf.matmul(t1, t2), 1)                             #(Nt, 1)

            t4 = tf.expand_dims(tf.squeeze(tf.matmul(t0,t2)),1) #(Nt, 1, n_labels)
            vec = tf.squeeze(tf.matmul(t4, left), 1)             #(Nt, D)
            norms = tf.linalg.norm(t4, axis = 2)                #(Nt, 1)
            #the label tensor is on the left side of self.pos, index samples with self.pos-1\
            prediction = tf.squeeze(t4)/norms
            loss = 1/2 * tf.math.reduce_mean(tf.math.reduce_sum((tf.squeeze(t4)/norms - labels) ** 2, 1),0)
            grad = tf.math.reduce_mean(
                tf.reshape(batched_kronecker(batched_kronecker((-y + vec * factor /(tf.pow(norms, 2)))/norms,samples[:,:,self.pos - 1]), right),(Nt, Dl, d, Dr)),
                axis=0
            )
            return grad, loss, prediction
            
        elif self.pos < self.label_pos:
            #this is a bit of a hack; the user should make sure that he did a left-rgiht and a right-left sweep prior to optimization
            if self.pos > 0:
                Dl = self._tensors[self.pos - 1].shape[2]
            else:
                Dl = 1
            d = self._tensors[self.pos].shape[1]
            Dr = self._tensors[self.pos].shape[2] 
            #print('Dl,d,Dr: ',Dl, d, Dr)
            Nt, n_labels = labels.shape
            left = self.left_data_environment[self.pos]                    #(Nt, 1, D)
            right = self.right_data_environment[self.pos]                   #(Nt, D, n_labels)
            #print('right: ', right.shape)
            y = tf.squeeze(tf.matmul(right, tf.expand_dims(labels, 2)), 2)  #(Nt, D)
            #print('y: ', y.shape)
            
            t0 = misc_mps.ncon([left, self.centermatrix],[[-1, -2, 1],[1, -3]])   #(Nt, 1, D)
            t1 = self.add_layer(samples, self.pos, -1)                            #(Nt, D, n_labels)
            t2 = tf.matmul(t1, tf.expand_dims(labels, 2))                         #(Nt, D, 1)
           
            factor = tf.squeeze(tf.matmul(t0, t2), 1)                             #(Nt, 1)
            #print('factor: ', factor.shape)
            t3 = tf.expand_dims(tf.squeeze(tf.matmul(t0, t1),1),2)                #(Nt, n_labels, 1)
            #print('t3: ', t3.shape, right.shape)
            norms = tf.linalg.norm(t3, axis = 1)                                  #(Nt, 1)
            #print('norms: ', norms.shape)
            vec = tf.squeeze(tf.matmul(right, t3), 2)                             #(Nt, D)
            #print('vec: ', vec.shape)
            #print('ampels: ', samples[:,:,self.pos].shape)
            #print('bla: ', bla.shape)
            #test = batched_kronecker(batched_kronecker(tf.squeeze(left, 1), samples[:, :, self.pos]),(-y + vec * factor /(tf.pow(norms, 2)))/norms)
            prediction = tf.squeeze(t3)/norms
            loss = 1/2 * tf.math.reduce_mean(tf.math.reduce_sum((tf.squeeze(t3)/norms - labels) ** 2, 1),0)            
            #print('test: ', test.shape)
            grad = tf.math.reduce_mean(
                tf.reshape(batched_kronecker(batched_kronecker(tf.squeeze(left, 1), samples[:, :, self.pos]),(-y + vec * factor /(tf.pow(norms, 2)))/norms), (Nt, Dl, d, Dr)),
                axis=0)
            return grad, loss, prediction
            

        
    def do_one_site_step(self,
                         samples,
                         labels,
                         direction,
                         learning_rate=1E-5):

        """
        learning rate should be positive
        """
        if direction in ('right','r'):
            assert (self.pos < len(self)) #the case of self.pos == len(self) is currently not implemented for self.one_site_gradient
                   
            gradient, loss, prediction = self.one_site_gradient(samples, labels)
            gradient_norm = tf.linalg.norm(gradient)
            gradient /= gradient_norm #probably not strictly neccessary
            temp = self.get_tensor(self.pos)
            self._tensors[self.pos] = (temp - learning_rate * gradient)
            self.mat = tf.eye(temp.shape[0], dtype = self.dtype) #the centermatrix was absorbed into the mps tensor; reset it to 11
            self.position(self.pos + 1)
            self.add_layer(samples, self.pos - 1, direction=1)

        #at this point self.pos could be len(self), depending on how often the user called do_one_site_step
        if self.pos == len(self):
            self.position(self.pos - 1)
            self.add_layer(samples, self.pos, direction=-1)
            
        if direction in ('l','left'):
            assert(self.pos > 0)
            gradient, loss, prediction = self.one_site_gradient(samples, labels)
            gradient_norm = tf.linalg.norm(gradient)            
            gradient /= gradient_norm
            #merge the label-tensor into the mps from the right
            temp = self.get_tensor(self.pos)
            self._tensors[self.pos] = (temp - learning_rate * gradient)
            self.mat = tf.eye(temp.shape[2], dtype = self.dtype)#the centermatrix was absorbed into the mps tensor; reset it to 11, and replace it one site to the right            
            self.pos += 1
            self.position(self.pos - 2)            
            self.add_layer(samples, self.pos + 1, direction=-1)
            self.add_layer(samples, self.pos, direction=-1)            
            
        return loss, gradient, gradient_norm


    def left_right_sweep_simple(self,
                                samples,
                                labels,
                                learning_rate,
                                t0=0.0,
                                n1=None):
        """
        minimizes the  cost function by sweeping from left to right
        Args:
            samples (tf.Tensor of shape (Nt, d, N) ):   the samples; Nt is the number of training samples
                                                        d is the embedding dimension, and N is the systemsize
            labels (tf.Tensor of shape (Nt, n_labels)): the one-hot encoded labels; Nt is the number of training samples
                                                        n_labels is the number of labels
            learning_rate (float):
            numpy_svd (bool):
            D (int or None):
            trunc_thresh (float):
        Returns:
            list of scalar tf.Tensor: the losses
            list of scalar tf.Tensor: the training accuracies     
        """
        losses = []
        train_accuracies = []
        if not n1:
            n1 = len(self)
        ground_truth = tf.argmax(labels,  axis=1)
        while self.pos < n1 - 1:
            loss, gradient, gradient_norm = self.do_one_site_step(samples, 
                                                                  labels,
                                                                  learning_rate=learning_rate, 
                                                                  direction='r')
            acc = self.accuracy(samples, labels)
            train_accuracies.append(acc)
            losses.append(loss)
            #print(tf.math.reduce_min(gradient),tf.math.reduce_max(gradient))
            stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f, D=%i, ||gradient|| + %.4f, time %0.2f" % (self.pos, 
                                                                                                                                  np.real(loss), 
                                                                                                                                  np.imag(loss), 
                                                                                                                                  learning_rate, 
                                                                                                                                  acc,
                                                                                                                                  self._tensors[self.pos].shape[0],
                                                                                                                                  gradient_norm,
                                                                                                                                  time.time() - t0))
            stdout.flush()
            
        with open(self.name + 'data_.pickle', 'wb') as f:
            pickle.dump({'losses':losses, 'accuracies': train_accuracies},f)
        return losses, train_accuracies
    
    def left_right_sweep_label(self,
                               samples,
                               labels,
                               learning_rate,
                               t0=0.0,
                               D=None,
                               n1=None):
        """
        minimizes the  cost function by sweeping from left to right
        Args:
            samples (tf.Tensor of shape (Nt, d, N) ):   the samples; Nt is the number of training samples
                                                        d is the embedding dimension, and N is the systemsize
            labels (tf.Tensor of shape (Nt, n_labels)): the one-hot encoded labels; Nt is the number of training samples
                                                        n_labels is the number of labels
            learning_rate (float):
            numpy_svd (bool):
            D (int or None):
            trunc_thresh (float):
        Returns:
            list of scalar tf.Tensor: the losses
            list of scalar tf.Tensor: the training accuracies     
        """
        losses = []
        train_accuracies = []
        if not n1:
            n1 = len(self)
        ground_truth = tf.argmax(labels,  axis=1)
        assert(self.label_pos == 0)
        assert(self.pos == 0)
        while self.pos < n1 - 1:
            loss, gradient, gradient_norm = self.do_one_site_step(samples, 
                                                                  labels,
                                                                  learning_rate=learning_rate, 
                                                                  direction='r')
            self.label_position(self.label_pos + 1, D=D)
            self.add_layer(samples, self.pos - 1, 1)
            acc = self.accuracy(samples, labels)
            train_accuracies.append(acc)
            losses.append(loss)
            #print(tf.math.reduce_min(gradient),tf.math.reduce_max(gradient))
            stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f, D=%i, ||gradient|| + %.4f, time %0.2f" % (self.pos, 
                                                                                                                                  np.real(loss), 
                                                                                                                                  np.imag(loss), 
                                                                                                                                  learning_rate, 
                                                                                                                                  acc,
                                                                                                                                  self._tensors[self.pos].shape[0],
                                                                                                                                  gradient_norm,
                                                                                                                                  time.time() - t0))
            stdout.flush()
            
        with open(self.name + 'data_.pickle', 'wb') as f:
            pickle.dump({'losses':losses, 'accuracies': train_accuracies},f)
        return losses, train_accuracies            
                
    def right_left_sweep_simple(self,
                                samples,
                                labels,
                                learning_rate,
                                t0=0.0,
                                n0=0):
        """
        minimizes the  cost function by sweeping from right to left
        Args:
            samples (tf.Tensor of shape (Nt, d, N) ):   the samples; Nt is the number of training samples
                                                        d is the embedding dimension, and N is the systemsize
            labels (tf.Tensor of shape (Nt, n_labels)): the one-hot encoded labels; Nt is the number of training samples
                                                        n_labels is the number of labels
            learning_rate (float):
            numpy_svd (bool):
            D (int or None):
            trunc_thresh (float):
        Returns:
            list of scalar tf.Tensor: the losses
            list of scalar tf.Tensor: the training accuracies     
        """
        
        losses = []
        train_accuracies = []

        ground_truth = tf.argmax(labels,  axis=1)
        while self.pos > n0 + 1:
            loss, gradient, gradient_norm = self.do_one_site_step(samples, 
                                                                  labels,
                                                                  learning_rate=learning_rate, 
                                                                  direction='l')
            acc = self.accuracy(samples, labels)            
            train_accuracies.append(acc) 
            losses.append(loss)
            #print(tf.math.reduce_min(gradient).n,tf.math.reduce_max(gradient))            
            stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f, D=%i, ||gradient|| + %.4f, time %0.2f" % (self.pos, 
                                                                                                                                  np.real(loss), 
                                                                                                                                  np.imag(loss), 
                                                                                                                                  learning_rate, 
                                                                                                                                  acc,
                                                                                                                                  self._tensors[self.pos].shape[0],
                                                                                                                                  gradient_norm,
                                                                                                                                  time.time() - t0))
            stdout.flush()
        with open(self.name + 'data_.pickle', 'wb') as f:
            pickle.dump({'losses':losses, 'accuracies': train_accuracies},f)
                
        return losses, train_accuracies
    
    def right_left_sweep_label(self,
                               samples,
                               labels,
                               learning_rate,
                               D=None,
                               t0=0.0,
                               n0=0):
        """
        minimizes the  cost function by sweeping from right to left
        Args:
            samples (tf.Tensor of shape (Nt, d, N) ):   the samples; Nt is the number of training samples
                                                        d is the embedding dimension, and N is the systemsize
            labels (tf.Tensor of shape (Nt, n_labels)): the one-hot encoded labels; Nt is the number of training samples
                                                        n_labels is the number of labels
            learning_rate (float):
            numpy_svd (bool):
            D (int or None):
            trunc_thresh (float):
        Returns:
            list of scalar tf.Tensor: the losses
            list of scalar tf.Tensor: the training accuracies     
        """
        
        losses = []
        train_accuracies = []

        ground_truth = tf.argmax(labels,  axis=1)
        while self.pos >= n0 + 1:
            loss, gradient, gradient_norm = self.do_one_site_step(samples, 
                                                                  labels,
                                                                  learning_rate=learning_rate, 
                                                                  direction='l')
            self.label_position(self.label_pos - 1, D=D)
            self.add_layer(samples, self.pos + 1, -1)
            
            acc = self.accuracy(samples, labels)            
            train_accuracies.append(acc) 
            losses.append(loss)
            #print(tf.math.reduce_min(gradient).n,tf.math.reduce_max(gradient))            
            stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f, D=%i, ||gradient|| + %.4f, time %0.2f" % (self.pos, 
                                                                                                                                  np.real(loss), 
                                                                                                                                  np.imag(loss), 
                                                                                                                                  learning_rate, 
                                                                                                                                  acc,
                                                                                                                                  self._tensors[self.pos].shape[0],
                                                                                                                                  gradient_norm,
                                                                                                                                  time.time() - t0))
            stdout.flush()
        with open(self.name + 'data_.pickle', 'wb') as f:
            pickle.dump({'losses':losses, 'accuracies': train_accuracies},f)
                
        return losses, train_accuracies


    def optimize(self, samples, labels, learning_rate, num_sweeps, numpy_svd=False,
                 D=None,
                 factor=1.5,
                 lower_lr_bound=1E-10,
                 max_local_steps=1,
                 max_steps_per_lr=4,                                 
                 trunc_thresh=1E-8,
                 loss_thresh=1E100,
                 t0=0.0):
        
        losses, accuracies = [],[]        
        for sweep in range(num_sweeps):
            
            loss, accs = self.left_right_sweep_simple(samples, labels, learning_rate,
                                                      t0=t0)
            losses.extend(loss)
            accuracies.extend(accs)
            
            loss, accs = self.right_left_sweep_simple(samples, labels, learning_rate,
                                                      t0=t0)
            
            losses.extend(loss)
            accuracies.extend(accs)
        return losses, accuracies

    
    
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



def shuffle(X, new_order=None):
    out = np.copy(X)
    if not new_order:
        new_order = list(range(X.shape[1]))
        np.random.shuffle(new_order)
    for t in range(X.shape[0]):
        out[t,:] =  X[t,new_order]
    return out, new_order


def one_hot_encoder(nb, Y):
    return np.eye(nb)[Y].astype(np.int32)[0]

    
def generate_mapped_MNIST_batches(X,Y,n_batches,which='one_hot',scaling=1.0, shuffle_pixels=False):
    """
    X is an M by N matrix, where M is the number of samples and N is the number of features 
    (N= 28*28 for the MNIST data set) Y are the labels 
    returns [f_1(x),f_2(x),Y], where f_1=cos(pi*x*256/(2*255)) and f_2=sin(pi*x*256/(2*255))
    pass this to calculateGradient
    """
    if shuffle_pixels:
        X = shuffle(X)
    def smooth_labels(labels,sig, n_labels):
        x = np.arange(n_labels)
        out = np.zeros((len(labels),n_labels))
        for n in range(len(labels)):
            out[n,:] = np.exp(-(x - labels[n])**2/(2*sig**2))
            Z = np.linalg.norm(out[n,:])
            out[n,:] /= Z
        return out

    batch_size = X.shape[0] // n_batches
    nb = len(np.unique(Y))
    if which == 'one_hot':
        y_one_hot = [scaling * np.eye(nb)[np.array([Y[n*batch_size:(n+1)*batch_size]])].squeeze().astype(np.int32)  
                     for n in range(n_batches)]
    elif which =='gaussian':
        sig = 10/8
        n_labels = len(np.unique(Y))        
        y_one_hot = [scaling * smooth_labels(Y[n*batch_size:(n+1)*batch_size],sig,n_labels) 
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
                 
    samples = [np.array([np.cos((X[n,:]*256)/255*math.pi/2),np.sin((X[n,:]*256)/255*math.pi/2)])
                    for n in range(X.shape[0])]    
    return samples,y_one_hot

 
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





    
