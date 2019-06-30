import sys
from sys import stdout
import tensorflow as tf
import numpy as np
import itertools
import math
import experiments.MPS.misc_mps as misc_mps
import tensornetwork as tn
import time
import pickle


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
    @tf.contrib.eager.defun    
    def shift_right(label_tensor, tensor):
        D = tensor.shape[1]
        net = tn.TensorNetwork()
        l_node =  net.add_node(label_tensor)
        t_node = net.add_node(tensor)
        e = net.connect(l_node[2], t_node[0])
        left_edges = [l_node[0],  t_node[2]]
        right_edges = [l_node[1], t_node[1]]        
        res = net.contract(e)
        u_node, s_node, v_node, _ = net.split_node_full_svd(res, left_edges, right_edges, max_singular_values=D)
        Z = tf.linalg.norm(s_node.tensor)
        s_node.tensor /= Z
        out = u_node.reorder_axes([0, 2, 1])
        label_tensor = net.contract(s_node[1])
        return out.tensor , label_tensor.tensor
    
    @staticmethod
    @tf.contrib.eager.defun
    def shift_left(tensor, label_tensor):
        D = tensor.shape[0]
        net = tn.TensorNetwork()
        l_node =  net.add_node(label_tensor)
        t_node = net.add_node(tensor)
        e = net.connect(l_node[0], t_node[1])
        left_edges = [t_node[0],  l_node[1]]
        right_edges = [l_node[2], t_node[2]]        
        res = net.contract(e)
        u_node, s_node, v_node, _ = net.split_node_full_svd(res, left_edges, right_edges, max_singular_values=D)
        Z = tf.linalg.norm(s_node.tensor)
        s_node.tensor /= Z
        label_tensor = net.contract(s_node[0])
        return label_tensor.tensor, v_node.tensor


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
        
    def position(self, bond):
        """
        
        """
        if bond == self.pos:
            return
        if bond > self.pos:

            for n in range(self._position, min(bond,len(self))):
                self.tensors[n],  self.label_tensor = self.shift_right(self.label_tensor, self.tensors[n])
            self._position = min(bond,len(self))
        if bond < self._position:
            for n in range(self._position - 1, max(-1,bond - 1), -1):
                self.label_tensor, self.tensors[n] = self.shift_left(self.tensors[n], self.label_tensor)
            self._position = max(0,bond)

        for s in reversed(range(self.pos)):
            if s in self.right_data_environment:
                del self.right_data_environment[s]
        for s in range(self.pos, len(self) + 1):
            if s in self.left_data_environment:            
                del self.left_data_environment[s]
            
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
            self.left_data_environment[site + 1] = tf.expand_dims(tf.squeeze(self.left_data_environment[site + 1])/out,1)

                
        if direction in (-1,'r','right'):
            assert(self.pos <= site)
            if site == (len(self) - 1):
                self.right_data_environment[site - 1] = misc_mps.ncon([self.tensors[site],embedded_data[:,:,site]],
                                         [[-2,-3,1],[-1,1]]) #has shape (Nt, D[-1], 1)
                
            else:
                #first contract the image pixels into the mps tensor
                tensor = misc_mps.ncon([self.tensors[site],embedded_data[:,:,site]],
                                   [[-2,-3,1],[-1, 1]]) #has shape (Nt, D[site], D[site+1])
                #use tf.matmul with broadcasting to multiply the right-next vectors
                #has shape (Nt, D[site],1)
                self.right_data_environment[site - 1] = tf.matmul(tensor, self.right_data_environment[site]) 
            out = tf.linalg.norm(self.right_data_environment[site - 1], axis=1) #get the norm of each row
            self.right_data_environment[site - 1]= tf.expand_dims(tf.squeeze(self.right_data_environment[site - 1])/out,2)
            
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
                left_envs = tf.squeeze(self.left_data_environment[self.pos]) #dummy index can be dropped 
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
                left_envs = tf.squeeze(self.left_data_environment[self.pos - 1]) #dummy index can be dropped 
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
            print(t6.shape)
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
                left_env = tf.squeeze(self.left_data_environment[self.pos])
                Dl = left_env.shape[1]
            else:
                left_env = tf.ones(shape=(Nt,1),  dtype=self.dtype) #FIXME: that's overkill; a single vector is enough        
                Dl = 1
            if self.pos < len(self) - 1:
                right_env = tf.squeeze(self.right_data_environment[self.pos])
                Dr = right_env.shape[1]
            else:
                right_env = tf.ones(shape=(Nt, 1), dtype=self.dtype)
                Dr = 1
            dl = embedded_data.shape[1]
            predict, norms = self.predict(embedded_data, which) #predictions are already normalized 
            #print(predict.dot(predict.T))
            #predict.shape is (Nt,n_labels)
            #each row in `predict` gets multiplied by `temp`.
            #`temp` is a set of `Nt` numbers, obtained from contracting the label vector for sample `n`
            #with the prediction vector for sample `n`.
            temp = tf.squeeze(tf.matmul(np.expand_dims(predict,1), np.expand_dims(labels, 2)),1)      
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
                left_env = tf.squeeze(self.left_data_environment[self.pos - 1])
                Dl = left_env.shape[1]
            else:
                left_env = tf.ones(shape=(Nt,1), dtype=self.dtype) #FIXME: that's overkill; a single vector is enough        
                Dl = 1
            if self.pos < len(self):
                right_env = tf.squeeze(self.right_data_environment[self.pos - 1])
                Dr = right_env.shape[1]
            else:
                right_env = tf.ones(shape=(Nt, 1), dtype=self.dtype)
                Dr = 1
            dl = embedded_data.shape[1]
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
    
    
    def do_one_site_step(self, embedded_data, labels, direction, learning_rate=-1E-5):
        """
        learning rate should be negative
        """
        if direction in ('right','r'):
            D = self.tensors[self.pos].shape[1]
            gradient, loss = self.one_site_gradient(embedded_data, labels, which='r')
            gradient /= tf.linalg.norm(gradient)
            #merge the label-tensor into the mps from the left
            temp = self.get_central_one_site_tensor(which = 'r')
            temp = (temp + learning_rate * gradient)


            net = tn.TensorNetwork()
            node = net.add_node(temp)
            left_edges = [node[0],  node[3]]
            right_edges = [node[1], node[2]]
            u_node, s_node, v_node, _ = net.split_node_full_svd(node, left_edges, right_edges, max_singular_values=D)
            Z = tf.linalg.norm(s_node.tensor)
            s_node.tensor /= Z
            out = u_node.reorder_axes([0, 2, 1])
            label_tensor = net.contract(s_node[1])
            

            self.tensors[self.pos] = out.tensor
            self.label_tensor = label_tensor.tensor
            self._position += 1
            self.add_layer(embedded_data, self.pos - 1, direction=1)
            
        if direction in ('l','left'):
            gradient, loss = self.one_site_gradient(embedded_data, labels, which='l')
            gradient /= np.linalg.norm(gradient)
            #merge the label-tensor into the mps from the right
            temp = self.get_central_one_site_tensor(which = 'l')
            temp = (temp + learning_rate * gradient)

            D = self.tensors[self.pos - 1].shape[0]

            net = tn.TensorNetwork()
            node = net.add_node(temp)
            left_edges = [node[0],  node[1]]
            right_edges = [node[2], node[3]]
            
            u_node, s_node, v_node, _ = net.split_node_full_svd(node, left_edges, right_edges, max_singular_values=D)
            Z = tf.linalg.norm(s_node.tensor)
            s_node.tensor /= Z
            label_tensor = net.contract(s_node[0])

            self.tensors[self.pos - 1] = v_node.tensor
            self.label_tensor = label_tensor.tensor
            self._position -= 1
            self.add_layer(embedded_data, self.pos, direction=-1)
            
        return loss, gradient
    
    def optimize(self, samples, labels, learning_rates, num_sweeps, n0=0):
        losses = []
        train_accuracies = []
        walltimes = [] 
        self.position(n0)
        self.compute_data_environments(samples)
        ground_truth = tf.argmax(labels,  axis=1)
        old_loss = 1E200
        t1 = time.time()
        for sweep in range(num_sweeps):
            if sweep < len(learning_rates):
                lr = learning_rates[sweep]
            else:
                lr = learning_rates[-1]
            for site in range(n0, len(self) - n0):
                loss, gradient = self.do_one_site_step(samples, 
                                                       labels, learning_rate=lr, 
                                                       direction='r')
                if self.pos < len(self):
                    prediction = tf.argmax(self.predict(samples, which='r', debug=False)[0], 1)
                else:
                    prediction = tf.argmax(self.predict(samples, which='l', debug=False)[0], 1)
                correct = np.sum(prediction.numpy()==ground_truth.numpy())
                train_accuracies.append(correct)
                losses.append(loss)
                stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f" % (site, 
                                                                                               np.real(loss), 
                                                                                               np.imag(loss), 
                                                                                               lr, 
                                                                                               correct/labels.shape[0]))
                stdout.flush()
            walltimes.append(time.time() - t1)
            t1 = time.time()
            for site in reversed(range(n0, len(self) - n0)):
                loss, gradient = self.do_one_site_step(samples, 
                                                       labels, learning_rate=lr, 
                                                       direction='l')
                if self.pos < len(self):
                    prediction = tf.argmax(self.predict(samples, which='r', debug=False)[0], 1)
                else:
                    prediction = tf.argmax(self.predict(samples, which='l', debug=False)[0], 1)
                    
                correct = np.sum(prediction.numpy()==ground_truth.numpy())
                train_accuracies.append(correct)                
                losses.append(loss)                
                stdout.write("\rsite %i, loss = %0.8f , %0.8f, lr = %0.8f, accuracy: %0.8f" % (site, 
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





    
