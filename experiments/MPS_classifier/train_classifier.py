import tensorflow as tf
tf.enable_v2_behavior()
import tensornetwork as tn
import experiments.MPS_classifier.MPSMNIST as mm
import datetime
import pickle
if __name__ == "__main__":
    n_batches = 1
    D = 60
    n_sweeps = 10
    dtype = tf.float64
    train,valid,test = mm.load_MNIST('data')
    data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=n_batches)
    samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
    labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))
    valid_data, valid_labels = mm.generate_mapped_MNIST_batches(valid[0], valid[1],n_batches=1)
    N = samples.shape[2]
    
    mps = mm.MPSClassifier.eye(d = [2]*N, D=[D]*N, num_labels=10, scaling=1.0, noise=1E-4, dtype=dtype)
    mps.position(0)
    n0 = 0
    lrs = [-0.00001, -0.000001, -0.000001] 
    today = str(datetime.date.today())   
    losses, accuracies, walltimes = mps.optimize(samples, labels, learning_rates=lrs, num_sweeps=n_sweeps, n0=n0)
    with open(today + '_MPS_MNIST_classifier_D{}.pickle'.format(D), 'wb') as f:
        pickle.dump(mps, f)
        
    
    
