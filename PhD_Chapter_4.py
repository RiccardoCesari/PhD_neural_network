## Used for linear case, with chi, with only one dimensional Y


import time, math
import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.stats import multivariate_normal as normal

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.compat.v1 import random_normal_initializer as norm_init
from tensorflow.compat.v1 import random_uniform_initializer as unif_init
from tensorflow.compat.v1 import constant_initializer as const_init


class SolveNN(object):
    """The fully-connected neural network model."""
    def __init__(self, sess):
        self.sess = sess
        # PDE parameters
        self.d = 4
        self.dimW = 2
        self.Xinit = np.array([0.5, 0, 0, 20])
        self.T = 0.5
        
        self.chi = 0.3
        self.phi = [0.003, 0.06, 0.06]
        self.k = 0.0
        self.sigma = [0.1, 0.1]
        self.k_bar = 0.2
        self.eta = 0.003
        self.rho = -0.4
        self.mu_1 = 0.0
        
        # Algorithm parameters
        self.n_time = 50
        self.n_neuron = [self.dimW, self.dimW + 3, self.dimW + 3, self.dimW]
        self.test_size = 400
        self.n_maxstep = 40000
        self.n_displaystep = 500
        self.Yinit = [self.Xinit[3] / 2, self.Xinit[3] * 2]
        self.Zinit = [-0.5, 0.5]
        
        self.learning_rate = 5e-3
        self.batch_size = 256
        self.valid_size = 1024
        print('The condition for >= 0 is: ', self.phi[2] * math.exp(1 + self.phi[1] / self.k_bar) - self.k_bar * math.exp((self.sigma[1] ** 2) / (2 * self.k_bar) + self.rho * self.sigma[0] * self.sigma[1] / self.k_bar))
        time.sleep(4)
        
        # Constants and Variables
        self.chi_on_eta = self.chi / self.eta
        self.phi_on_eta = [phi / self.eta for phi in self.phi]
        self.k_bar_on_eta = self.k_bar / self.eta
        self.term_end = self.sigma[1]**2 / (4 * self.eta) + self.rho * self.sigma[0] * self.sigma[1] / (2 * self.eta)
        self.h = (self.T + 0.0) / self.n_time
        self.sqrth = math.sqrt(self.h)
        self.t_stamp = np.arange(0, self.n_time + 1) * self.h
        self._extra_train_ops = []

    
    def test(self):
        dW_test, X_test = self.sample_path(self.test_size)
        feed_dict_test = {self.dW: dW_test, self.X: X_test, self.is_training: False}
        self.print_table(feed_dict_test, final_step=True)

    def train(self):
        self.start_time = time.time()
        self.display_time = self.t_bd + self.start_time
        # Training Operations
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(1), trainable=False, dtype=tf.int32)
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainable_vars)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        apply_op = optimizer.apply_gradients(zip(grads, trainable_vars), global_step=self.global_step)
        train_ops = [apply_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)
        self.loss_history = []
        self.init_history = []
        
        # Validation Config
        dW_valid, X_valid = self.sample_path(self.valid_size)
        feed_dict_valid = {self.dW: dW_valid, self.X: X_valid, self.is_training: False}
        
        # Initialization
        step = 1
        print("--------> call initializer")
        self.sess.run(tf.global_variables_initializer())
        temp_loss = self.sess.run(self.loss, feed_dict=feed_dict_valid)
        temp_init = self.Y0.eval()
        self.loss_history.append({'step':0, 'loss':temp_loss})
        self.init_history.append(temp_init)
        self.call_display(0, temp_loss, temp_init)
        
        # Start Iteration
        for _ in range(self.n_maxstep + 1):
            step = self.sess.run(self.global_step)
            dW_train, X_train = self.sample_path(self.batch_size)
            self.sess.run(self.train_op, feed_dict={self.dW: dW_train, self.X: X_train, self.is_training: True})
            if step % self.n_displaystep == 0:
                temp_loss = self.sess.run(self.loss, feed_dict=feed_dict_valid)
                temp_init = self.Y0.eval()
                self.loss_history.append({'step':step, 'loss':temp_loss})
                self.init_history.append(temp_init)
                self.call_display(step, temp_loss, temp_init)
                self.print_table(feed_dict_valid)
            step += 1
        self.end_time = time.time()
        print("running time: %.3f s" % (self.end_time + self.display_time))


    def build(self):
        start_time = time.time()
        # stacking subnetworks
        self.print_X = []
        self.print_Y = []
        self.print_Z = []
        self.dW = tf.placeholder(tf.float64, [None, self.dimW, self.n_time], name='dW')
        self.X = tf.placeholder(tf.float64, [None, self.d], name='X')
        self.is_training = tf.placeholder(tf.bool)
        self.Y0 = tf.Variable(tf.random_uniform([1], minval=self.Yinit[0], maxval=self.Yinit[1], dtype=tf.float64))
        self.Z0 = tf.Variable(tf.random_uniform([1, self.dimW], minval=self.Zinit[0], maxval=self.Zinit[1], dtype=tf.float64))
        self.allones = tf.ones(tf.stack([tf.shape(self.dW)[0], 1]), dtype=tf.float64)
        self.allones2 = tf.ones(tf.stack([tf.shape(self.dW)[0], 1]), dtype=tf.float64)
        self.eye_batch = tf.eye(tf.shape(self.dW)[0], dtype=tf.float64)
        Y = self.allones * self.Y0
        X = tf.matmul(self.eye_batch, self.X)
        print(Y)
        Z = tf.matmul(self.allones2, self.Z0)
        print(Z)
        with tf.variable_scope('forward'):
            for t in range(0, self.n_time  - 1):
                self.print_Y.append(Y[:,0])
                self.print_X.append(X[:,:])
                self.print_Z.append(Z[:,:])
                Y = Y - self.f_tf(X, Y, Z) * self.h
                Y = tf.where(tf.math.greater(X[:,3], 0.0), Y + tf.reduce_sum(Z * self.dW[:, :, t], 1, keep_dims=True), Y)
                X = X + self.mu_tf(X, Y, self.dW[:,:,t]) * self.h + self.sigma_tf(X, self.dW[:, :, t])
                X = self.check_pos(X)
                Z = self._one_time_net(X[:,:2], str(t + 1)) / self.dimW
                
            # terminal time
            print(Y)
            self.print_Y.append(Y[:,0])
            self.print_X.append(X[:,:])
            self.print_Z.append(Z[:,:])
            Y = Y - self.f_tf(X, Y, Z) * self.h
            Y = tf.where(tf.math.greater(X[:,3], 0.0), Y + tf.reduce_sum(Z * self.dW[:, :, t], 1, keep_dims=True), Y)
            X = X + self.mu_tf(X, Y, self.dW[:,:,t]) * self.h + self.sigma_tf(X, self.dW[:, :, t])
            term_delta = Y - self.g_tf(X)
            X = X + self.term_condition(X)
            self.print_Y.append(Y[:,0])
            self.print_X.append(X[:,:])
            self.print_Z.append(Z[:,:])
            term_delta_norm = tf.norm(term_delta, axis=1)
            self.clipped_delta = tf.clip_by_value(term_delta_norm, -5000.0, 5000.0)
            self.loss = tf.reduce_mean(tf.math.square(self.clipped_delta))
        self.t_bd = time.time() - start_time


    def call_display(self, step_, loss_, init_):
        print("step: %5u ===> loss= %.4e, Y0= %.4e, " % (step_, loss_, init_) + "runtime: %4u" % (time.time() + self.display_time))

    def sample_path(self, n_size):
        dW_ = np.zeros([n_size, self.dimW, self.n_time])
        X_ = np.zeros([n_size, self.d])
        for i_ in range(n_size):
            X_[i_, :] = self.Xinit
        for i in range(self.n_time):
            dW_[:, :, i] = np.reshape(normal.rvs(mean=np.zeros(self.dimW), cov=1, size=n_size) * self.sqrth, (n_size, self.dimW))
        return dW_, X_
    
        
    def explicit_optimal_c(self, t, A, eps, Q):
        def g_3_old_paper(t):
            gamma = math.sqrt(self.phi[0] / self.eta)
            beta = math.sqrt(self.phi[0] * self.eta)
            return beta * (np.exp(2 * gamma * t) * (beta - self.chi) - math.exp(2 * gamma * self.T) * (beta + self.chi)) / (np.exp(2 * gamma * t) * (beta - self.chi) + math.exp(2 * gamma * self.T) * (beta + self.chi))

        def g_hat_old_paper(r_vec, t):
            num_steps = 400
            s_vec = np.linspace(t, r_vec[0], num_steps)
            h = (t - r_vec[0]) / (num_steps - 1)
            final_vec = [h * np.sum(g_3_old_paper(s_vec))]
            for i in range(len(r_vec) - 1):
                s_vec = np.linspace(r_vec[i], r_vec[i + 1], num_steps)
                h = (r_vec[i + 1] - r_vec[i]) / (num_steps - 1)
                final_vec.append(final_vec[-1] + h * np.sum(g_3_old_paper(s_vec)))
            return np.exp(np.array(final_vec) / self.eta)

        def mu_bar_old_paper(t):
            return np.exp(-self.k_bar * t)

        def sigma_bar_sq_old_paper(t):
            return (self.sigma[1] ** 2) * (1 - np.exp(-2 * self.k_bar * t)) / (2 * self.k_bar)

        def g_2_old_paper(t, eps):
            num_steps = 400
            h = (self.T - t) / num_steps
            r_vec = np.linspace(t, self.T, num_steps - 1)
            integ_1 = np.sum(g_hat_old_paper(r_vec, t))
            integ_2 = np.sum(g_hat_old_paper(r_vec, t) * np.exp(mu_bar_old_paper(r_vec - t) * eps + sigma_bar_sq_old_paper(r_vec - t)  / 2 + self.rho * self.sigma[0] * self.sigma[1] * (1 - mu_bar_old_paper(r_vec - t)) / self.k_bar) * (self.k_bar * mu_bar_old_paper(r_vec - t) * eps + self.k_bar * sigma_bar_sq_old_paper(r_vec - t) - (self.sigma[1] ** 2) / 2 - self.rho * self.sigma[0] * self.sigma[1] * mu_bar_old_paper(r_vec - t) + self.phi[1]))
            return 1 - self.phi[2] * math.exp(-eps) * integ_1 * h - math.exp(-eps) * integ_2 * h
        
        if Q <= 0:
            return 0
        else:
            return (A * math.exp(eps) * (1 - g_2_old_paper(t, eps)) - 2 * Q * g_3_old_paper(t)) / (2 * self.eta)
    
    def mu_tf(self, X, Y, dW):
        return tf.where(tf.math.greater(X[:,3], 0.0), tf.transpose(tf.stack([self.mu_1 * X[:,0], -self.k_bar * X[:,1], self.indic_f(Y[:,0]) * (X[:,0] * tf.math.exp(X[:,1]) - self.eta * self.indic_f(Y[:,0])), - self.indic_f(Y[:,0])])), X * 0.0)
    
    def sigma_tf(self, X, dW):
        return tf.transpose(tf.stack([self.sigma[0] * X[:,0] * dW[:,0], self.sigma[1] * (self.rho * dW[:,0] + math.sqrt(1-self.rho**2) * dW[:,1]), X[:,0] * 0, X[:,0] * 0]))

    def f_tf(self, X, Y, Z):
        # nonlinear term
        return tf.reshape(tf.where(tf.math.greater(X[:,3], 0.0), self.phi_on_eta[0] * X[:,3] + 0.5 * X[:,0] * (self.phi_on_eta[1] + self.phi_on_eta[2] * tf.math.exp(X[:,1])) - X[:,0] * tf.math.exp(X[:,1]) * (- self.k_bar_on_eta * X[:,1] / 2 + self.term_end), X[:,3] * 0.0), [tf.shape(self.dW)[0],1])

    def g_tf(self, X):
        # terminal conditions
        return tf.reshape(tf.where(tf.math.greater(X[:,3], 0.0), self.chi_on_eta * X[:,3], X[:,0] * 0), [tf.shape(self.dW)[0],1])
        
    def term_condition(self, X):
        return tf.transpose(tf.stack([X[:,3] * 0.0, X[:,3] * 0.0, X[:,3] * (X[:,0] * tf.math.exp(X[:,1]) - self.chi * X[:,3]), -self.indic_f(X[:,3])]))
    
    def indic_f(self, X):
        return tf.where(tf.math.greater(X, 0.0), X, tf.math.abs(X) * 0.0)
    
    def check_pos(self, X):
        return tf.transpose(tf.stack([X[:,0], X[:,1], X[:,2], self.indic_f(X[:,3])]))

    def _one_time_net(self, x, name):
        with tf.variable_scope(name):
            x_norm = self._batch_norm(x, name='layer0_normal')
            layer1 = self._one_layer(x_norm, self.n_neuron[1], name='layer1')
            layer2 = self._one_layer(layer1, self.n_neuron[2], name='layer2')
            z = self._one_layer(layer2, self.n_neuron[3], name='final')
        return z

    def _batch_norm(self, x, name):
        """Batch Normalization"""
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]
            beta = tf.get_variable('beta', params_shape, tf.float64,
                                   norm_init(0.0, stddev=0.1, dtype=tf.float64))
            gamma = tf.get_variable('gamma', params_shape, tf.float64,
                                    unif_init(0.1, 0.5, dtype=tf.float64))
            mv_mean = tf.get_variable('moving_mean', params_shape, tf.float64,
                                      const_init(0.0, tf.float64), trainable=False)
            mv_var = tf.get_variable('moving_variance', params_shape, tf.float64,
                                     const_init(1.0, tf.float64), trainable=False)
            # Training Ops
            mean, variance = tf.nn.moments(x, [0], name='moments')
            hoge = assign_moving_average(mv_mean, mean, 0.99)
            piyo = assign_moving_average(mv_var, variance, 0.99)
            self._extra_train_ops.extend([hoge, piyo])
            mean, variance = control_flow_ops.cond(self.is_training,
                                                   lambda : (mean, variance),
                                                   lambda : (mv_mean, mv_var))
            y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-6)
            y.set_shape(x.get_shape())
            return y


    def _one_layer(self, input_, out_size, f_activation=tf.nn.relu, std=5.0, name='linear'):
        with tf.variable_scope(name):
            shape = input_.get_shape().as_list()
            w = tf.get_variable('Matrix', [shape[1], out_size], tf.float64, norm_init(stddev=std / np.sqrt(shape[1] + out_size)))
            hidden = tf.matmul(input_, w)
            hidden_bn = self._batch_norm(hidden, name='normal')
        if f_activation != None:
            return f_activation(hidden_bn)
        else:
            return hidden_bn
    
    def print_table(self, feed_dict, final_step=False):
        ## used to print tables and charts with results
        pd.options.display.float_format = '{:,.2f}'.format
        self.array_control_X = self.sess.run(self.print_X, feed_dict=feed_dict)
        self.array_control_Y = self.sess.run(self.print_Y, feed_dict=feed_dict)
        self.array_control_Z = self.sess.run(self.print_Z, feed_dict=feed_dict)
        X_out = self.array_control_X
        Y_out = self.array_control_Y
        Z_out = self.array_control_Z
        array_print_table = [0, 10, 20, 30]
        table_total = []
        if final_step:
            fig = go.Figure()
            range_loop = range(len(X_out[0]))
        else:
            range_loop = [0, 10]
        for print_i in range_loop:
            Q_opt = self.Xinit[3]
            M_opt = self.Xinit[2]
            table_array = []
            for i in range(self.n_time + 1):
                A = X_out[i][print_i, 0]
                eps = X_out[i][print_i, 1]
                S = A * math.exp(eps)
                M = X_out[i][print_i, 2]
                Q = X_out[i][print_i, 3]
                Z_11 = Z_out[i][print_i, 0]
                Z_12 = Z_out[i][print_i, 1]
                if i == self.n_time:
                    c = 0
                    c_opt = 0
                    M_opt += Q_opt * (S - self.chi * Q_opt) * int(Q_opt > 0)
                    Q_opt = 0
                else:
                    c = Y_out[i][print_i]
                    c_opt = self.explicit_optimal_c(self.t_stamp[i], A, eps, Q_opt)
                table_array.append({'test_n':print_i, 't':i, 'c':c, 'A':A, 'eps':eps, 'M':M, 'Q':Q, 'S':S, 'c_opt':c_opt, 'Q_opt': Q_opt, 'M_opt': M_opt, 'Z_11': Z_11, 'Z_12': Z_12})
                Q_opt -= c_opt * self.h
                M_opt += c_opt * (S - self.eta * c_opt) * self.h
            table_to_print = pd.DataFrame(table_array)
            table_total += table_array
            if print_i in array_print_table:
                print(table_to_print[['t', 'A', 'eps', 'S', 'c', 'c_opt', 'Q', 'Q_opt', 'M', 'M_opt', 'Z_11', 'Z_12']])
                if final_step:
                    fig.add_trace(go.Scatter(x=table_to_print['t'], y=table_to_print['c'], mode='lines', name='c' + str(print_i)))
                    fig.add_trace(go.Scatter(x=table_to_print['t'], y=table_to_print['c_opt'], mode='lines', name='c_opt' + str(print_i)))
        if final_step:
            fig.show()
            
            fig2 = go.Figure()
            table_loss = pd.DataFrame(self.loss_history)
            fig2.add_trace(go.Scatter(x=table_loss['step'], y=np.log10(table_loss['loss']), mode='lines', name='error'))
            fig2.show()
            
            table_to_print = pd.DataFrame(table_total)
            table_loss.to_csv('table_loss_BSDE_18.csv')
            self.do_statistics(table_to_print)
    
    def do_statistics(self, table):
        table['abs_diff_M'] = np.abs(table['M'] - table['M_opt'])
        table['abs_diff_Q'] = np.abs(table['Q'] - table['Q_opt'])
        table['abs_diff_c'] = np.abs(table['c'] - table['c_opt'])
        table['rel_diff_M'] = np.abs((table['M'] - table['M_opt']) / table['M'])
        table['rel_diff_Q'] = np.abs((table['Q'] - table['Q_opt']) / table['Q'])
        table['rel_diff_c'] = np.abs((table['c'] - table['c_opt']) / table['c'])
        table.to_csv('result_BSDE_18.csv')
        table_grouped = table.groupby(['t']).agg({'abs_diff_M':['mean', 'std', 'median', q1, q2], 'abs_diff_Q':['mean', 'std', 'median', q1, q2], 'abs_diff_c':['mean', 'std', 'median', q1, q2], 'rel_diff_M':['mean', 'std', 'median', q1, q2], 'rel_diff_Q':['mean', 'std', 'median', q1, q2], 'rel_diff_c':['mean', 'std', 'median', q1, q2]})
        print(table_grouped)
        table_grouped.to_csv('result_grouped_BSDE_18.csv')

GLOBAL_RANDOM_SEED = 1

def q1(x):
    return x.quantile(0.75)

def q2(x):
    return x.quantile(0.25)

def main():
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.set_random_seed(GLOBAL_RANDOM_SEED)
        model = SolveNN(sess)
        model.build()
        model.train()
        print("------> Testing")
        model.test()
        

if __name__ == '__main__':
    np.random.seed(GLOBAL_RANDOM_SEED)
    main()