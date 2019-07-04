import tensorflow as tf
tf.enable_v2_behavior()
import tensornetwork as tn
import experiments.MPS_classifier.MPSMNIST as mm
import datetime
import pickle
import time
if __name__ == "__main__":
    numpy_svd  = False
    factor = 1.5
    n_stpcnt = 20
    lower_bound = 1E-9
    max_local_steps =  2
    trunc_thresh = 1E-8
    n_batches = 50
    n_sweeps = 10
    dtype = tf.float64
    train,valid,test = mm.load_MNIST('data')
    data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=n_batches)
    samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
    labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))
    valid_data, valid_labels = mm.generate_mapped_MNIST_batches(valid[0], valid[1],n_batches=1)
    N = samples.shape[2]
    
    mps = mm.MPSClassifier.eye(d = [2]*N, D=[2]*N, num_labels=10, scaling=1.0, noise=1E-6, dtype=dtype)
    mps.position(0, trunc_thresh=trunc_thresh)
    mps.compute_data_environments(samples)    
    lr = 1E-8
    loss_thresh=1E100
    today = str(datetime.date.today())
    losses,  accuracies = [], []
    t0 = time.time()    
    l, a = mps.left_right_sweep_simple(samples,labels,learning_rate=1E-6,numpy_svd=numpy_svd,
                                       trunc_thresh=1E-8,t0=t0)
    losses.extend(l)
    accuracies.extend(a)
    
    l, a = mps.right_left_sweep_simple(samples,labels,learning_rate=1E-8,numpy_svd=numpy_svd,
                                       trunc_thresh=1E-8, loss_thresh=loss_thresh,t0=t0)
    losses.extend(l)
    accuracies.extend(a)

    for n in range(n_sweeps -1):
        l, a = mps.left_right_sweep_simple(samples,labels,learning_rate=1E-8,numpy_svd=numpy_svd,
                                           trunc_thresh=trunc_thresh, loss_thresh=loss_thresh,t0=t0)
        losses.estend(l)
        accuracies.extend(a)
        
        l, a = mps.right_left_sweep_simple(samples,labels,learning_rate=1E-8,numpy_svd=numpy_svd,
                                           trunc_thresh=trunc_thresh, loss_thresh=loss_thresh, t0=t0)    
        losses.extend(l)
        accuracies.extend(a)
                                                        
    
    with open(today + '_MPS_MNIST_classifier_D{}.pickle'.format(D), 'wb') as f:
        pickle.dump(mps, f)
        
    
    
