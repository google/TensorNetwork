import sys
from sys import stdout
import numpy as np
import experiments.MPS.misc_mps as misc_mps
import itertools
import experiments.MPS_classifier.MPSMNISTbkp as mm
import pytest
import tensorflow as tf
tf.enable_v2_behavior()


def test_predict():
    dtype  = tf.float64
    train,valid,test = mm.load_MNIST('data')
    data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=500)
    samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
    labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))
    N = samples.shape[2]
    mps = mm.MPSClassifier.random(d = [2]*N, D=[11]*N, num_labels=10, scaling=1.0, dtype = dtype)
    mps.position(0)
    mps.position(len(mps)//2)
    mps.compute_data_environments(samples)
    
    def test(left_envs, label_tensor, right_envs, bottom):
        left_envs = tf.squeeze(left_envs)
        right_envs = tf.squeeze(right_envs)
        bottom = tf.squeeze(bottom)
        temp = np.zeros(label_tensor.shape[1])
        for t in range(left_envs.shape[0]):
            temp2 = misc_mps.ncon([left_envs[t,:], bottom[t,:], right_envs[t,:], label_tensor],[[1],[2],[3],[1,-1,3,2]])
            temp2 /= tf.linalg.norm(temp2)
            temp += temp2
        return temp
    
    t1, norms = mps.predict(embedded_data=samples, which='r')
    t2, norms = mps.predict(embedded_data=samples, which='l')
    t3 = test(mps.left_data_environment[mps.pos], mps.get_central_one_site_tensor(which='r'),
              mps.right_data_environment[mps.pos], samples[:,:,mps.pos])
    
    t4 = test(mps.left_data_environment[mps.pos - 1], mps.get_central_one_site_tensor(which='l'),
              mps.right_data_environment[mps.pos - 1], samples[:,:,mps.pos - 1])    
    np.testing.assert_allclose(t3, np.sum(t1, 0))
    np.testing.assert_allclose(t4, np.sum(t2, 0))

    
# def test_data_envs():
#     dtype  = tf.float64
#     train,valid,test = mm.load_MNIST('data')
#     data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=500)
#     samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
#     labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))

#     N = samples.shape[2]
#     mps = mm.MPSClassifier.random(d = [2]*N, D=[11]*N, num_labels=10, scaling=1.0, dtype=dtype)
#     mps.position(0)
#     mps.position(len(mps)//2)
#     mps.compute_data_environments(samples)
#     Nt = samples.shape[0]
    
#     lefts = np.zeros((Nt, mps.Dl[mps.pos]))
#     rights = np.zeros((Nt, mps.Dr[mps.pos]))
#     for t in range(samples.shape[0]):
#         left = np.ones((1))
#         right = np.ones((1))
#         for n in range(mps.pos):
#             left = misc_mps.ncon([left, samples[t,:,n], mps.tensors[n]],[[1], [2],[1,-1,2]])
#             left/=np.linalg.norm(left)
#         lefts[t,:] = left
#         for n in reversed(range(mps.pos, len(mps))):
#             right = misc_mps.ncon([mps.tensors[n],samples[t,:,n], right],[[-1,1,2], [2], [1]]) 
#             right/=np.linalg.norm(right)
#         rights[t,:] = right
#     np.testing.assert_allclose(lefts, mps.left_data_environment[mps.pos][:,0,:])
#     np.testing.assert_allclose(rights, mps.right_data_environment[mps.pos-1][:,:, 0])

@pytest.mark.parametrize("Nt", [100])
@pytest.mark.parametrize("d", [4])
@pytest.mark.parametrize("dtype", [tf.float64])
def test_batched_kronecker(Nt, d, dtype):
    a = tf.random_uniform(shape=[Nt, d], dtype=dtype)
    b = tf.random_uniform(shape=[Nt, d], dtype=dtype)
    c = tf.random_uniform(shape=[Nt, d], dtype=dtype)
    w = tf.random_uniform(shape=[Nt, d], dtype=dtype)
    out = mm.batched_kronecker(mm.batched_kronecker(mm.batched_kronecker(a,b),c), w)
    out2 = np.zeros(out.get_shape()).astype(dtype.as_numpy_dtype)
    for t in range(Nt):
        out2[t,:] = np.kron(np.kron(np.kron(a[t,:],b[t,:]),c[t,:]), w[t,:])

    np.testing.assert_allclose(out,out2)




    
@pytest.mark.parametrize("which", ['r','l'])    
def test_gradient(which):
    dtype  = tf.float64
    train,valid,test = mm.load_MNIST('data')
    data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=500)
    samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
    labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))

    N = samples.shape[2]
    mps = mm.MPSClassifier.random(d = [2]*N, D=[11]*N, num_labels=10, scaling=1.0, dtype=dtype)
    mps.position(0)
    mps.position(len(mps)//2)
    mps.compute_data_environments(samples)
    if which == 'r':
        gradient, loss = mps.one_site_gradient(samples, labels, which=which)
        n_labels = labels.shape[1]
        left = tf.squeeze(mps.left_data_environment[mps.pos])
        bottom = samples[:,:,mps.pos]
        predict, norms = mps.predict(samples, which=which)
        right = tf.squeeze(mps.right_data_environment[mps.pos])
        Dl, Dr, dl = left.shape[1], right.shape[1], bottom.shape[1]
        out = np.zeros((Dl, n_labels, Dr, dl))
        for t in range(labels.shape[0]):
            top = (tf.tensordot(predict[t,:], labels[t, :], ([0],[0])) * predict[t, :] - labels[t, :])/norms[t,0]
            out += misc_mps.ncon([left[t,:], top, right[t,:], bottom[t,:]],[[-1],[-2],[-3],[-4]])
        np.testing.assert_allclose(gradient,out)
        
    if which == 'l':
        gradient, loss = mps.one_site_gradient(samples, labels, which=which)
        n_labels = labels.shape[1]
        left = tf.squeeze(mps.left_data_environment[mps.pos-1])
        bottom = samples[:,:,mps.pos-1]
        predict, norms = mps.predict(samples, which=which)
        right = tf.squeeze(mps.right_data_environment[mps.pos-1])
        Dl, Dr, dl = left.shape[1], right.shape[1], bottom.shape[1]
        out = np.zeros((Dl, n_labels, Dr, dl))
        for t in range(labels.shape[0]):
            top = (tf.tensordot(predict[t,:],labels[t, :], ([0],[0])) * predict[t, :] - labels[t, :])/norms[t,0]
            out += misc_mps.ncon([left[t,:], top, right[t,:], bottom[t,:]],[[-1],[-2],[-3],[-4]])
        np.testing.assert_allclose(gradient,out)        

