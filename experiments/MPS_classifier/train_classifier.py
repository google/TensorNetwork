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
    trunc_thresh = 1E-8
    D = 100
    loss_thresh=0.1
    today = str(datetime.date.today())
    mps = mm.MPSClassifier.eye(d = [2]*N, D=[4]*N, num_labels=10, scaling=1.0, noise=1E-4, dtype=dtype)
    mps.position(0, trunc_thresh=trunc_thresh, D=4)
    losses,  accuracies = [], []    
    for nb1, nb2 in [(100,50),(20, 10), (10,10)]:
        data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=nb1)
        samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
        labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))
        mps.compute_data_environments(samples)    
    
        
        t0 = time.time()
        l, a = mps.left_right_sweep_simple(samples,labels,learning_rate=lr,numpy_svd=numpy_svd,
                                           trunc_thresh=trunc_thresh,D=D,t0=t0,
                                           max_local_steps=4, loss_thresh=loss_thresh,lower_lr_bound=1E-8)
        losses.extend(l)
        accuracies.extend(a)
        data, labs = mm.generate_mapped_MNIST_batches(train[0],train[1],n_batches=nb2)
        samples = tf.convert_to_tensor(data[0].astype(dtype.as_numpy_dtype))
        labels = tf.convert_to_tensor(labs[0].astype(dtype.as_numpy_dtype))
        mps.compute_data_environments(samples)    
        
        l, a = mps.right_left_sweep_simple(samples,labels,learning_rate=lr,numpy_svd=numpy_svd,
                                           trunc_thresh=trunc_thresh, t0=t0, D=D,
                                           max_local_steps=4, loss_thresh=loss_thresh,lower_lr_bound=1E-8)        
        losses.extend(l)
        accuracies.extend(a)
    
        # for n in range(n_sweeps -1):
        #     l, a = mps.left_right_sweep_simple(samples,labels,learning_rate=lr,numpy_svd=numpy_svd,
        #                                        trunc_thresh=trunc_thresh, loss_thresh=loss_thresh,t0=t0, D=D)
        #     losses.estend(l)
        #     accuracies.extend(a)
            
        #     l, a = mps.right_left_sweep_simple(samples,labels,learning_rate=lr,numpy_svd=numpy_svd,
        #                                        trunc_thresh=trunc_thresh, loss_thresh=loss_thresh, t0=t0, D=D)    
        #     losses.extend(l)
        #     accuracies.extend(a)
                                                            
    
        with open(today + '_MPS_MNIST_classifier_D{}.pickle'.format(D), 'wb') as f:
            pickle.dump(mps, f)
        
    
    
