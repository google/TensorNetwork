import sys
from sys import stdout
import numpy as np
import experiments.MPS.misc_mps as misc_mps
import itertools
import experiments.MPS_classifier.MPSMNIST as mm
import pytest
import tensorflow as tf
tf.enable_v2_behavior()


def test_predict():
    dtype  = tf.float64
    N = 196
    x_train = np.random.randn(100,N)
    y_train = np.random.randint(0,10,N).astype(np.int32)
    data, labs = mm.generate_mapped_MNIST_batches(x_train,y_train,n_batches=1)
    samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
    labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))
    mps = mm.MPSClassifier.random(ds = [2]*N, D=7, num_labels=10,label_position = N//2,
                                  scaling=1.0, name='test',dtype=dtype)

    sample = tf.expand_dims(samples[0,:,:],0)
    label = tf.expand_dims(labels[0,:],0)    

    mps.position(len(mps))
    mps.position(0)    
    def predictor(mps, sample, pos):
        mps.position(pos)
        mps.compute_data_environments(sample)
        left = tf.ones(shape = [1], dtype = mps.dtype)
        for site in range(mps.label_pos):
            left =  misc_mps.ncon([left, mps.get_tensor(site), sample[0,:,site]],[[1], [1,2,-1], [2]])
            Z = tf.linalg.norm(left)
            left /= Z
        right= tf.ones(shape = [1], dtype = mps.dtype)
        for site in reversed(range(mps.label_pos + 1, len(mps))):
            #print(site)
            right = misc_mps.ncon([mps.get_tensor(site), sample[0,:,site - 1], right],[[-1,1,2], [1], [2]])
            Z = tf.linalg.norm(right)
            right /= Z
        pred= misc_mps.ncon([left, mps.get_tensor(mps.label_pos), right],[[1], [1, -1, 2], [2]])
        return pred / tf.linalg.norm(pred)            

    a = predictor(mps,sample, mps.label_pos-40).numpy()
    b = predictor(mps,sample,mps.label_pos).numpy()
    c = predictor(mps,sample,mps.label_pos + 40).numpy()

    mps.position(mps.label_pos-40)
    mps.compute_data_environments(sample)
    p, n = mps.predict(sample)
    d = np.squeeze(p.numpy())
    mps.position(mps.label_pos)
    mps.compute_data_environments(sample)
    p, n = mps.predict(sample)
    e = np.squeeze(p.numpy())    

    
    mps.position(mps.label_pos + 40)
    mps.compute_data_environments(sample)
    p, n = mps.predict(sample)
    f = np.squeeze(p.numpy())        
    
    np.testing.assert_allclose(a,b)
    np.testing.assert_allclose(a,c)    
    np.testing.assert_allclose(a,d)
    np.testing.assert_allclose(a,e)
    np.testing.assert_allclose(a,f)            

    

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




    
