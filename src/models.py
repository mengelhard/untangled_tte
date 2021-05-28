import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers


class DNMC(Model):
    
    def __init__(self,
                 phi_layer_sizes=[256,], psi_layer_sizes=[256,], omega_layer_sizes=[256,],
                 e_layer_sizes=[256,], t_layer_sizes=[256,], c_layer_sizes=[256,],
                 importance_weights=[1., 1.],
                 include_censoring_density=True,
                 n_bins=50,
                 activation='relu',
                 rep_activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-8):
        
        super(DNMC, self).__init__()
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.include_censoring_density = include_censoring_density
        self.n_bins = n_bins
        self.activation = activation
        self.tol = tol
        
        self.phi_layers = [self.dense(ls, activation=rep_activation) for ls in phi_layer_sizes]
        self.psi_layers = [self.dense(ls, activation=rep_activation) for ls in psi_layer_sizes]
        self.omega_layers = [self.dense(ls, activation=rep_activation) for ls in omega_layer_sizes]
        
        self.e_layers = [self.dense(ls) for ls in e_layer_sizes] + [Dense(1, activation='sigmoid')]
        self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [Dense(n_bins, activation='softmax')]
        
        self.phi_model = Sequential(self.phi_layers)
        self.psi_model = Sequential(self.psi_layers)
        self.omega_model = Sequential(self.omega_layers)
        
        self.e_model = Sequential(self.e_layers)
        self.t_model = Sequential(self.t_layers)
        
        if include_censoring_density:
            self.c_layers = [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins, activation='softmax')]
            self.c_model = Sequential(self.c_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=regularizers.l2(self.lr))
        
        return(layer)
        
    
    def forward_pass(self, x):
        
        self.phi = self.phi_model(x)
        self.psi = self.psi_model(x)
        self.omega = self.omega_model(x)
        
        self.e_pred = tf.squeeze(self.e_model(tf.concat([self.phi, self.psi], axis=-1)), axis=1)
        self.t_pred = self.t_model(tf.concat([self.psi, self.omega], axis=-1))
        
        if self.include_censoring_density:
            self.c_pred = self.c_model(tf.concat([self.psi, self.omega], axis=-1))
            return self.e_pred, self.t_pred, self.c_pred

        else:
            return self.e_pred, self.t_pred
    
    
    def call(self, x):
        return self.forward_pass(x)
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, y, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, y, s))
        l = nll + self.ld * tf.cast(self.mmd(x, s), dtype=tf.float32)
        l += tf.reduce_sum(self.losses)
        
        return l, nll
    
    
    def nll(self, x, y, s):
        
        yt = tf.cast(y, dtype=tf.float32)
        
        if self.include_censoring_density:
            
            e_pred, t_pred, c_pred = self.forward_pass(x)
            
            fc = tf.reduce_sum(yt * c_pred, axis=1) + self.tol
            Fc = tf.reduce_sum(yt * self._survival_from_density(c_pred), axis=1) + self.tol
        
        else:

            e_pred, t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = tf.reduce_sum(yt * self._survival_from_density(t_pred), axis=1) + self.tol
        
        ll1 = tf.math.log(e_pred) + tf.math.log(ft) + tf.math.log(Fc)
        ll2 = tf.math.log(1 - e_pred * (1 - Ft)) + tf.math.log(fc)
            
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll
    
    
    def mmd(self, x, s, beta=1.):
        
        x0 = tf.boolean_mask(x, s == 0, axis=0)
        x1 = tf.boolean_mask(x, s == 1, axis=0)

        x0x0 = self._gaussian_kernel(x0, x0, beta)
        x0x1 = self._gaussian_kernel(x0, x1, beta)
        x1x1 = self._gaussian_kernel(x1, x1, beta)
        
        return tf.reduce_mean(x0x0) - 2. * tf.reduce_mean(x0x1) + tf.reduce_mean(x1x1)


    def _gaussian_kernel(self, x1, x2, beta=1.):
        return tf.exp(-1. * beta * tf.reduce_sum((x1[:, tf.newaxis, :] - x2[tf.newaxis, :, :]) ** 2, axis=-1))


    def _survival_from_density(self, f):
    	return tf.math.cumsum(f, reverse=True, axis=1)


class NMC(Model):
    
    def __init__(self,
                 e_layer_sizes=[256, 256], t_layer_sizes=[256, 256], c_layer_sizes=[256, 256],
                 importance_weights=[1., 1.],
                 include_censoring_density=True,
                 n_bins=50,
                 activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-3):
        
        super(NMC, self).__init__()
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.include_censoring_density = include_censoring_density
        self.n_bins = n_bins
        self.activation = activation
        self.tol = tol
        
        self.e_layers = [self.dense(ls) for ls in e_layer_sizes] + [Dense(1, activation='sigmoid')]
        self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [Dense(n_bins, activation='softmax')]
        
        self.e_model = Sequential(self.e_layers)
        self.t_model = Sequential(self.t_layers)
        
        if include_censoring_density:
            self.c_layers = [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins, activation='softmax')]
            self.c_model = Sequential(self.c_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=regularizers.l2(self.lr))
        
        return(layer)
        
    
    def forward_pass(self, x):
        
        self.e_pred = tf.squeeze(self.e_model(x), axis=1)
        self.t_pred = self.t_model(x)

        if self.include_censoring_density:
            self.c_pred = self.c_model(x)
            return self.e_pred, self.t_pred, self.c_pred

        else:
            return self.e_pred, self.t_pred
    
    
    def call(self, x):
        return self.forward_pass(x)
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, y, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, y, s))
        l = nll + tf.reduce_sum(self.losses)
        
        return l, nll
    

    def nll(self, x, y, s):
        
        yt = tf.cast(y, dtype=tf.float32)
        
        if self.include_censoring_density:
            
            e_pred, t_pred, c_pred = self.forward_pass(x)
            
            fc = tf.reduce_sum(yt * c_pred, axis=1) + self.tol
            Fc = tf.reduce_sum(yt * self._survival_from_density(c_pred), axis=1) + self.tol
        
        else:

            e_pred, t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = tf.reduce_sum(yt * self._survival_from_density(t_pred), axis=1) + self.tol
        
        ll1 = tf.math.log(e_pred) + tf.math.log(ft) + tf.math.log(Fc)
        ll2 = tf.math.log(1 - e_pred * (1 - Ft)) + tf.math.log(fc)
            
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll


    def _survival_from_density(self, f):
    	return tf.math.cumsum(f, reverse=True, axis=1)


class NSurv(Model):
    
    def __init__(self,
                 t_layer_sizes=[256, 256], c_layer_sizes=[256, 256],
                 importance_weights=[1., 1.],
                 include_censoring_density=True,
                 n_bins=50,
                 activation='relu',
                 ld=1e-3, lr=1e-3, tol=1e-3):
        
        super(NSurv, self).__init__()
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.include_censoring_density = include_censoring_density
        self.n_bins = n_bins
        self.activation=activation
        self.tol = tol
        
        self.t_layers = [self.dense(ls) for ls in t_layer_sizes] + [Dense(n_bins, activation='softmax')]
        self.t_model = Sequential(self.t_layers)

        if self.include_censoring_density:
            self.c_layers = [self.dense(ls) for ls in c_layer_sizes] + [Dense(n_bins, activation='softmax')]
            self.c_model = Sequential(self.c_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=regularizers.l2(self.lr))
        
        return(layer)
        
    
    def forward_pass(self, x):
        
        self.t_pred = self.t_model(x)

        if self.include_censoring_density:
            self.c_pred = self.c_model(x)
            return self.t_pred, self.c_pred

        else:
            return self.t_pred
    
    
    def call(self, x):
        return self.forward_pass(x)
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, y, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, y, s))
        l = nll + tf.reduce_sum(self.losses)
        
        return l, nll
    
    
    def nll(self, x, y, s):
        
        yt = tf.cast(y, dtype=tf.float32)
        
        if self.include_censoring_density:
            
            t_pred, c_pred = self.forward_pass(x)
            
            fc = tf.reduce_sum(yt * c_pred, axis=1) + self.tol
            Fc = tf.reduce_sum(yt * self._survival_from_density(c_pred), axis=1) + self.tol
        
        else:

            t_pred = self.forward_pass(x)

            fc = 1.
            Fc = 1.

        ft = tf.reduce_sum(yt * t_pred, axis=1) + self.tol
        Ft = tf.reduce_sum(yt * self._survival_from_density(t_pred), axis=1) + self.tol
        
        ll1 = tf.math.log(ft) + tf.math.log(Fc)
        ll2 = tf.math.log(Ft) + tf.math.log(fc)
            
        ll = tf.cast(s, dtype=tf.float32) * ll1
        ll += tf.cast((1 - s), dtype=tf.float32) * ll2
        
        return -1 * ll


    def _survival_from_density(self, f):
    	return tf.math.cumsum(f, reverse=True, axis=1)


class MLP(Model):
    
    def __init__(self,
                 e_layer_sizes=[256, 256],
                 importance_weights=[1., 1.],
                 activation='relu',
                 ld=1e-3, lr=1e-3):
        
        super(MLP, self).__init__()
        
        self.ld = ld
        self.lr = lr
        self.w0 = tf.convert_to_tensor(importance_weights[0], dtype=tf.float32)
        self.w1 = tf.convert_to_tensor(importance_weights[1], dtype=tf.float32)
        self.activation=activation
        
        self.e_layers = [self.dense(ls) for ls in e_layer_sizes] + [Dense(1, activation='sigmoid')]
        self.e_model = Sequential(self.e_layers)
        
        
    def dense(self, layer_size, activation=None):

        if activation is None:
            activation = self.activation
        
        layer = Dense(
            layer_size,
            activation=activation,
            kernel_regularizer=regularizers.l2(self.lr),
            bias_regularizer=regularizers.l2(self.lr))
        
        return(layer)
        
    
    def forward_pass(self, x):
        self.e_pred = tf.squeeze(self.e_model(x), axis=1)
        return self.e_pred
    
    
    def call(self, x):
        return self.forward_pass(x)
    
    
    def iweights(self, s):
        return tf.cast(s, dtype=tf.float32) * self.w1 + tf.cast((1 - s), dtype=tf.float32) * self.w0

    
    def loss(self, x, y, s):
        
        nll = tf.reduce_mean(self.iweights(s) * self.nll(x, y, s))
        l = nll + tf.reduce_sum(self.losses)
        
        return l, nll
    
    
    def nll(self, x, y, s):
        
        e_pred = self.forward_pass(x)
        
        l1 = e_pred
        l2 = (1 - e_pred)
            
        ll = tf.cast(s, dtype=tf.float32) * tf.math.log(l1)
        ll += tf.cast((1 - s), dtype=tf.float32) * tf.math.log(l2)
        
        return -1 * ll


def discrete_ci(st, tt, tp):

	s_true = np.array(st).copy()
	t_true = np.array(tt).copy()
	t_pred = np.array(tp).copy()

	t_true_idx = np.argmax(t_true, axis=1)
	t_pred_cdf = np.cumsum(t_pred, axis=1)

	concordant = 0
	total = 0

	N = len(s_true)
	idx = np.arange(N)

	for i in range(N):

		if s_true[i] == 0:
			continue

		# time bucket of observation for i, then for all but i
		tti_idx = t_true_idx[i]
		tt_idx = t_true_idx[idx != i]

		# calculate predicted risk for i at the time of their event
		tpi = t_pred_cdf[i, tti_idx]

		# predicted risk at that time for all but i
		tp = t_pred_cdf[idx != tti_idx, tti_idx]

		total += np.sum(tti_idx < tt_idx) # observed in i first
		concordant += np.sum((tti_idx < tt_idx) * (tpi > tp)) # and i predicted as higher risk

	return concordant / total


