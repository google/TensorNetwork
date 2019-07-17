import tensorflow as tf
tf.enable_v2_behavior()
import tensornetwork as tn
import experiments.MPS_classifier.MPSMNIST as mm
import datetime
import pickle
import time
if __name__ == "__main__":
    numpy_svd  = False
    n_sweeps = 10
    dtype = tf.float64
    train, valid, test = mm.load_MNIST('data')
    train[0], new_order = mm.shuffle(train[0])
    valid[0], _ = mm.shuffle(valid[0],new_order)
    test[0], _ = mm.shuffle(test[0],new_order)
    N = 784
    lr = 1E-4
    D = 10
    loss_thresh=0.1
    today = str(datetime.date.today())
    mps = mm.MPSClassifier.eye(ds = [2]*N, D=D, num_labels=10, label_position=0,scaling=1.0, noise=1E-4, dtype=dtype)
    mps.position(0,  D=D)
    losses,  accuracies = [], []    
    for nb1, nb2 in [(100,50),(20, 10), (10,10)]:
        data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=nb1)
        samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
        labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))
        t0 = time.time()
        
        mps.label_position(0,D=D)
        mps.position(0)
        mps.compute_data_environments(samples)        
        l, a = mps.left_right_sweep_label(samples,labels,learning_rate=1E-3,D=D)

        mps.label_position(N//2, D=D)        
        mps.position(0)
        mps.compute_data_environments(samples)        
        l, a = mps.left_right_sweep_simple(samples,labels,learning_rate=lr, t0=t0)
        losses.extend(l)
        accuracies.extend(a)
        data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=nb2)
        samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
        labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))
        mps.compute_data_environments(samples)    
        
        l, a = mps.right_left_sweep_simple(samples,labels,learning_rate=lr,
                                           t0=t0)
                                           
        losses.extend(l)
        accuracies.extend(a)
    
        for n in range(n_sweeps -1):
            l, a = mps.left_right_sweep_simple(samples,labels,learning_rate=lr,t0=t0)
            losses.extend(l)
            accuracies.extend(a)
            
            l, a = mps.right_left_sweep_simple(samples,labels,learning_rate=lr,t0=t0)
            losses.extend(l)
            accuracies.extend(a)
                                                            
    
        with open(today + '_MPS_MNIST_classifier_D{}.pickle'.format(D), 'wb') as f:
            pickle.dump(mps, f)
        
    
    
