import tensorflow as tf
tf.enable_v2_behavior()
import tensornetwork as tn
import experiments.MPS_classifier.MPSMNIST as mm
import datetime
import pickle
if __name__ == "__main__":
    numpy_svd  =True
    factor = 1.5
    n_stpcnt = 20
    lower_bound = 1E-9
    max_local_steps =  2
    
    n_batches = 1
    D = 100
    n_sweeps = 20
    dtype = tf.float64
    train,valid,test = mm.load_MNIST('data')
    data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=n_batches)
    samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
    labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))
    valid_data, valid_labels = mm.generate_mapped_MNIST_batches(valid[0], valid[1],n_batches=1)
    N = samples.shape[2]
    
    mps = mm.MPSClassifier.eye(d = [2]*N, D=[D]*N, num_labels=10, scaling=1.0, noise=1E-4, dtype=dtype)
    mps.position(0, numpy_svd=numpy_svd)
    n0 =56
    lr = -1E-6
    today = str(datetime.date.today())
    losses, accuracies, walltimes = mps.optimize(samples, labels, learning_rate=lr, num_sweeps=n_sweeps, n0=n0, numpy_svd=numpy_svd,
                                                 factor=factor, n_stpcnt=n_stpcnt, lower_bound=lower_bound, max_local_steps=max_local_steps)
    with open(today + '_MPS_MNIST_classifier_D{}.pickle'.format(D), 'wb') as f:
        pickle.dump(mps, f)
        
    
    
