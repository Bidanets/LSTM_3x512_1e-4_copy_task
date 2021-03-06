# -*- coding: utf-8 -*-


from controller import *


class Feedforward(Controller):
    def __init__(self, inp_vector_size, batch_size, layer_sizes):
        self.inp_vector_size = inp_vector_size
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.out_vector_size = self.layer_sizes[-1]

        print("Controller initialization begin.")
        print("self.inp_vector_size: ", self.inp_vector_size)
        print("self.batch_size", self.batch_size)
        print("self.layer_sizes", self.layer_sizes)
        print("self.out_vector_size", self.out_vector_size)
        print("Controller initialization end.")


    def initial_state(self, batch_size = None):
        return tf.constant(0)

    def __call__(self, x, sequence_length):
        """
        Should get around to implementing it one day.


        :param x:
        :param sequence_length:
        :return:
        """
        raise NotImplementedError()

    def step(self, x, state, step):
        """
        Returns the output vector for just one time step
        
        :param x: vector representing input for one time step
        :parma state: not used in any way whatsoever
        :param step: current time step
        :return: output of feedforward network and tf.constant(0), because it doesn't have a state, (required by 
                tf.while_loop)
        """
        

        with tf.variable_scope("FF_step"):
            for layer_size in self.layer_sizes[:-1]:
                #x = tf.Print(x, [x[0][0], x[0][1], x[0][2]], "\nInput object from dataset and read vector (in cycle by layer_sizes): \n", summarize = 1)

                x = tf.layers.dense(x, layer_size, activation = tf.nn.relu)
                #x = tf.Print(x, [x[0][0], x[0][1], x[0][2]], "\nFirst hidden layer: \n", summarize = 1)

            x = tf.layers.dense(x, self.layer_sizes[-1], activation = tf.nn.relu)
            #x = tf.Print(x, [x[0][0], x[0][1], x[0][2]], "\nSecond hidden layer: \n", summarize = 1)


        return x, state

    def notify(self, states):
        """
        FF controller doesn't have an internal state
        :param states: 
        :return: 
        """
        pass
