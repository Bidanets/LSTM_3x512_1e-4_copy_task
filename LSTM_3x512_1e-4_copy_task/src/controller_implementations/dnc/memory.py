# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os
import time

class Memory:
    """
    Used by the DNC. It doesn't inherit Controller, but its functionality has some overlap.
    
    Controller  |  Memory
    -------------------------
    step()      |  step()
    __call__()  |  None
    
    Memory doesn't have the Controller's __call__() equivalent, because memory depends on DNC.
    
    """
    epsilon = tf.constant(1e-6, dtype = tf.float64) 
    max_outputs = 2

    def __init__(self, batch_size, controller_output_size, out_vector_size, mem_hp, initial_stddev = 0.1):     

        """
        :param batch_size: 
        :param controller_output_size: size of the controller output vector. Used for calculating the shape of the 
               interface weight matrix
        :param out_vector_size: size of the output of memory, after being passed through the output weight matrix
        :param mem_hp: hyperparameters for memory, object with attributes word_size, mem_size, num_read_heads 
        :param initial_stddev: 
        """

        print("Memory initialization:")

        self.batch_size = batch_size
        #print("Batch size: ", self.batch_size)

        self.controller_output_size = controller_output_size
        #print("Controller output size: ", self.controller_output_size)

        self.out_vector_size = out_vector_size
        #print("Out vector size: ", self.out_vector_size)

        self.memory_size = mem_hp.mem_size
        #print("Memory size: ", self.memory_size)

        self.word_size = mem_hp.word_size
        #print("Word size: ", self.word_size)

        self.num_read_heads = mem_hp.num_read_heads
        #print("Num read heads: ", self.num_read_heads)

        self.interface_vector_size = (self.word_size * self.num_read_heads) + \
                                     5 * self.num_read_heads + 3 * self.word_size + 3
        
        #print("Interface vector size: ", self.interface_vector_size)

        self.interface_weights = tf.Variable(
            tf.random_normal([self.controller_output_size, self.interface_vector_size], stddev = initial_stddev, dtype = tf.float64),
            name = "interface_weights")
        
        """
        self.interface_weights = tf.Print(self.interface_weights, [
            self.interface_weights[0, 0], self.interface_weights[0, 1], self.interface_weights[0, 2], self.interface_weights[0, 3], self.interface_weights[0, 4], self.interface_weights[0, 5], self.interface_weights[0, 6], self.interface_weights[0, 7], self.interface_weights[0, 8], self.interface_weights[0, 9], self.interface_weights[0, 10], self.interface_weights[0, 11],
            self.interface_weights[1, 0], self.interface_weights[1, 1], self.interface_weights[1, 2], self.interface_weights[1, 3], self.interface_weights[1, 4], self.interface_weights[1, 5], self.interface_weights[1, 6], self.interface_weights[1, 7], self.interface_weights[1, 8], self.interface_weights[1, 9], self.interface_weights[1, 10], self.interface_weights[1, 11],
            self.interface_weights[2, 0], self.interface_weights[2, 1], self.interface_weights[2, 2], self.interface_weights[2, 3], self.interface_weights[2, 4], self.interface_weights[2, 5], self.interface_weights[2, 6], self.interface_weights[2, 7], self.interface_weights[2, 8], self.interface_weights[2, 9], self.interface_weights[2, 10], self.interface_weights[2, 11],
        ], "Interface weights: ", summarize = 1)
        """

        self.output_weights = tf.Variable(
            tf.random_normal([self.num_read_heads, self.word_size, self.out_vector_size], stddev = initial_stddev, dtype = tf.float64),
            name = "output_weights")
 
        # Что такое output_weights? и out vector size?
        """
        self.output_weights = tf.Print(self.output_weights, [
            self.output_weights[0][0][0], self.output_weights[0][0][1]
            ], "\noutput_weights: \n", summarize = 1)
        """

        # Code below is a more-or-less pythonic way to process the individual interface parameters
        r, w = self.num_read_heads, self.word_size
        sizes = [r * w, r, w, 1, w, w, r, 1, 1, 3 * r]

        names = ["r_read_keys", "r_read_strengths", "write_key", "write_strength", "erase_vector", "write_vector",
                 "r_free_gates", "allocation_gate", "write_gate", "r_read_modes"]

        functions = [tf.identity, Memory.oneplus, tf.identity, Memory.oneplus, tf.nn.sigmoid, tf.identity,
                     tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid, self.reshape_and_softmax]

        indexes = [[sum(sizes[:i]), sum(sizes[:i + 1])] for i in range(len(sizes))]

        assert len(names) == len(sizes) == len(functions) == len(indexes)

        # Computing the interface dictionary from ordered lists of sizes, vector names and functions, as in the paper
        # self.split_interface is called later with the actual computed interface vector
        self.split_interface = lambda iv: {name: fn(iv[:, i[0]:i[1]]) for name, i, fn in
                                           zip(names, indexes, functions)}

    def init_memory(self, batch_size):
        """
        Returns the memory state for step 0. Used in DNC for the argument to tf.while_loop
        
        :return: 
        """
        read_weightings = tf.fill([batch_size, self.memory_size, self.num_read_heads], Memory.epsilon)

        # --------------------------------------------------------------------------------------------------------------
        #read_weightings = tf.Print(read_weightings, [read_weightings[0, 0, 0], read_weightings[0, 1, 0]], "\nRead weightings: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        write_weighting = tf.fill([batch_size, self.memory_size], Memory.epsilon, name = "Write_weighting")

        # --------------------------------------------------------------------------------------------------------------
        #write_weighting = tf.Print(write_weighting, [write_weighting[0, 0], write_weighting[0, 1]], "\nWrite weighting: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        precedence_weighting = tf.zeros([batch_size, self.memory_size], name = "Precedence_weighting", dtype = tf.float64)

        # --------------------------------------------------------------------------------------------------------------
        #precedence_weighting = tf.Print(precedence_weighting, [precedence_weighting[0, 0], precedence_weighting[0, 1]], 
        #"\nPrecedence weighting: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        m = tf.fill([batch_size, self.memory_size, self.word_size], Memory.epsilon)  # initial memory matrix

        # --------------------------------------------------------------------------------------------------------------
        #m = tf.Print(m, [m[0, 0, 0], m[0, 1, 0]], "\nm: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        usage_vector = tf.zeros([batch_size, self.memory_size], name = "Usage_vector", dtype = tf.float64)

        # --------------------------------------------------------------------------------------------------------------
        #usage_vector = tf.Print(usage_vector, [usage_vector[0, 0], usage_vector[0, 1]], "\nUsage vector: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        link_matrix = tf.zeros([batch_size, self.memory_size, self.memory_size], dtype = tf.float64)

        # --------------------------------------------------------------------------------------------------------------
        #link_matrix = tf.Print(link_matrix, [link_matrix[0, 0, 0], link_matrix[0, 0, 1], link_matrix[0, 1, 0], link_matrix[0, 1, 1]], 
        #"\nLink matrix: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        read_vectors = tf.fill([batch_size, self.num_read_heads, self.word_size], Memory.epsilon)

        # --------------------------------------------------------------------------------------------------------------
        #read_vectors = tf.Print(read_vectors, [read_vectors[0, 0, 0]], "\nRead vectors: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        return [read_weightings, write_weighting, usage_vector, precedence_weighting, m, link_matrix, read_vectors]

    def step(self, controller_output, memory_state):
        """
        One step of memory. Performs all the read and write operations and returns some useful tensors for visualization
        
        :param controller_output: used to create the interface vector which parametrizes memory operations
        :param memory_state: list of a bunch of useful memory tensors from previous step which are necessary for 
            the calculation of next step
        :return: output of the memory, all the updated memory states and a bunch of of other tensors for visualization
        """

        read_weightings, write_weighting, usage_vector, precedence_weighting, m, link_matrix, read_vectors = \
            memory_state

        print(controller_output)
        print(self.interface_weights)

        interface_vector = controller_output @ self.interface_weights  # shape [batch_size, interf_vector_size] = [batch size, lstm layer size] x [output size, interface vector size]

        interf = self.split_interface(interface_vector)

        interf["write_key"] = tf.expand_dims(interf["write_key"], dim = 1)

        # --------------------------------------------------------------------------------------------------------
        #interf["write_key"] = tf.Print(interf["write_key"], [interf["write_key"][0, 0, 0]], "\niterf_write_key: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------

        interf["r_read_keys"] = tf.reshape(interf["r_read_keys"], [-1, self.num_read_heads, self.word_size])

        # --------------------------------------------------------------------------------------------------------
        #interf["r_read_keys"] = tf.Print(interf["r_read_keys"], [interf["r_read_keys"][0, 0, 0]], "\nr_read_keys: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------


        # memory retention is of shape [batch_size, memory_size]
        memory_retention = tf.reduce_prod(1 - tf.einsum("br,bnr->bnr", interf["r_free_gates"], read_weightings), 2)

        # --------------------------------------------------------------------------------------------------------------
        #memory_retention = tf.Print(memory_retention, [memory_retention[0, 0], memory_retention[0, 1]], "\nMemory retention: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        # calculating the usage vector, which has some sort of short-term memory (вычисление вектора использования, который играет некоторую роль краткосрочной памяти)
        usage_vector = (usage_vector + write_weighting - usage_vector * write_weighting) * memory_retention

        # --------------------------------------------------------------------------------------------------------------
        #usage_vector = tf.Print(usage_vector, [usage_vector[0, 0], usage_vector[0, 1]], "\nUsage vector: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        allocation_weighting = self.calculate_allocation_weighting(usage_vector)

        # --------------------------------------------------------------------------------------------------------------
        #allocation_weighting = tf.Print(allocation_weighting, [allocation_weighting[0, 0], allocation_weighting[0, 1]], "\nAllocation weighting: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------
        

        # calculating write content weighting with memory from previous time step, shape [batch_size, memory_size, 1]
        # it represents a normalized probability distribution over the memory locations
        write_content_weighting = Memory.content_based_addressing(m, interf["write_key"], interf["write_strength"])

        # --------------------------------------------------------------------------------------------------------------
        #write_content_weighting = tf.Print(write_content_weighting, [write_content_weighting[0, 0], write_content_weighting[0, 1]], "\nWrite content weighting: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------
        

        # write weighting, shape [batch_size, memory_size], represents a normalized probability distribution over the
        # memory locations. Its sum is <= 1
        write_weighting = interf["write_gate"] * (interf["allocation_gate"] * allocation_weighting
                                                  + tf.einsum("bnr,bi->bn",
                                                              write_content_weighting,
                                                              (1 - interf["allocation_gate"])))

        # --------------------------------------------------------------------------------------------------------------
        #write_weighting = tf.Print(write_weighting, [write_weighting[0, 0], write_weighting[0, 1]], "\nWrite weighting: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        m = m * (1 - tf.einsum("bn,bw->bnw", write_weighting, interf["erase_vector"])) + \
            tf.einsum("bn,bw->bnw", write_weighting, interf["write_vector"])

        # --------------------------------------------------------------------------------------------------------------
        #m = tf.Print(m, [m[0, 0, 0], m[0, 1, 0]], 
        #"\nm: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        link_matrix = self.update_link_matrix(link_matrix, precedence_weighting, write_weighting)

        # --------------------------------------------------------------------------------------------------------------
        #link_matrix = tf.Print(link_matrix, [link_matrix[0, 0, 0], link_matrix[0, 0, 1], link_matrix[0, 1, 0], link_matrix[0, 1, 1]], 
        #"\nLink matrix: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        precedence_weighting = tf.einsum("bn,b->bn",
                                         precedence_weighting,
                                         (1 - tf.reduce_sum(write_weighting, axis = 1))) + write_weighting

        forwardw = tf.einsum("bmn,bnr->bmr", link_matrix, read_weightings)
        
        # --------------------------------------------------------------------------------------------------------------
        #forwardw = tf.Print(forwardw, [forwardw[0, 0], forwardw[0, 1]], "\nForwardw: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------
        
        
        backwardw = tf.einsum("bnm,bnr->bmr", link_matrix, read_weightings)

        # --------------------------------------------------------------------------------------------------------------
        #backwardw = tf.Print(backwardw, [backwardw[0, 0], backwardw[0, 1]], "\nBackwardw: \n", summarize = 1) 
        # --------------------------------------------------------------------------------------------------------------
        

        # calculating rcw with updated memory, shape [batch_size, memory_size, num_read_heads]
        # it represents a normalized probability distribution over the memory locations, for each read head
        read_content_weighting = Memory.content_based_addressing(m, interf["r_read_keys"],
                                                                 interf["r_read_strengths"])

        # --------------------------------------------------------------------------------------------------------------
        #read_content_weighting = tf.Print(read_content_weighting, [read_content_weighting[0, 0], read_content_weighting[0, 1]], "\nRead content weighting: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------
                                                    

        read_weightings = Memory.calculate_read_weightings(interf["r_read_modes"],
                                                           backwardw,
                                                           read_content_weighting,
                                                           forwardw)

        # --------------------------------------------------------------------------------------------------------------
        #read_weightings = tf.Print(read_weightings, [read_weightings[0, 0, 0], read_weightings[0, 1, 0]], "\nRead weightings: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------
         
        read_vectors = tf.einsum("bnw,bnr->brw", m, read_weightings)
        
        # --------------------------------------------------------------------------------------------------------------
        #read_vectors = tf.Print(read_vectors, [read_vectors[0, 0, 0]], "\nRead vectors: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------


        memory_output = tf.einsum("brw,rwo->bo", read_vectors, self.output_weights)

        """
        memory_output = tf.Print(memory_output, [
            memory_output[0][0], memory_output[0][1] 
        ], "memory_output: ", summarize = 1)
        """

        extra_visualization_info = [interf["r_read_modes"], interf["write_gate"], interf["allocation_gate"],
                                    interf["write_strength"], interf["r_read_strengths"], interf["erase_vector"],
                                    interf["write_vector"], forwardw, backwardw, interf["r_free_gates"],
                                    interf["r_read_keys"]]

        memory_state = [read_weightings,
                        write_weighting,
                        usage_vector,
                        precedence_weighting,
                        m,
                        link_matrix,
                        read_vectors]

        return memory_output, memory_state, extra_visualization_info

    @staticmethod
    def calculate_read_weightings(r_read_modes, backwardw, read_content_weighting, forwardw):
        return tf.einsum("brs,bnrs->bnr", r_read_modes, tf.stack([backwardw, read_content_weighting, forwardw], axis = 3))

    def calculate_allocation_weighting(self, usage_vector):
        """
        :param: usage vector: tensor of shape [batch_size, memory_size]
        :return: allocation tensor of shape [batch_size, memory_size]
        """

        usage_vector = Memory.epsilon + (1 - Memory.epsilon) * usage_vector

        #usage_vector = tf.Print(usage_vector, [usage_vector[0][0], usage_vector[0][1]], "\nusage_vector_in_calculate_allocation_weighting_func: \n", summarize = 1)

        # We're sorting the "-self.usage_vector" because top_k returns highest values and we need the lowest
        highest_usage, inverse_indices = tf.nn.top_k(-usage_vector, k = self.memory_size)

        #highest_usage = tf.Print(highest_usage, [highest_usage[0][0], highest_usage[0][1]], "\nhighest_usage_in_calculate_allocation_weighting_func: \n", summarize = 1)

        #inverse_indices = tf.Print(inverse_indices, [inverse_indices[0][0], inverse_indices[0][1]], "\ninverse_indices_in_calculate_allocation_weighting_func: \n", summarize = 1)

        lowest_usage = -highest_usage

        #lowest_usage = tf.Print(lowest_usage, [lowest_usage[0][0], lowest_usage[0][1]], "\nlowest_usage_in_calculate_allocation_weighting_func: \n", summarize = 1)

        allocation_scrambled = (1 - lowest_usage) * tf.cumprod(lowest_usage, axis = 1, exclusive = True)

        #allocation_scrambled = tf.Print(allocation_scrambled, [allocation_scrambled[0][0], allocation_scrambled[0][1]],
        #                                "\nallocation_scrambled: \n", summarize = 1)

        # allocation is not in the correct order. alloation[i] contains the sorted[i] value
        # reversing the already inversed indices for each batch
        indices = tf.stack([tf.invert_permutation(batch_indices) for batch_indices in tf.unstack(inverse_indices)])

        #indices = tf.Print(indices, [indices[0][0], indices[0][1]], "\nindices: \n", summarize = 1)


        allocation = tf.stack([tf.gather(mem, ind)
                               for mem, ind in
                               zip(tf.unstack(allocation_scrambled), tf.unstack(indices))])

        #allocation = tf.Print(allocation, [allocation[0][0], allocation[0][1]],
        #                        "\nallocation: \n", summarize = 1)


        return allocation

    def update_link_matrix(self, link_matrix_old, precedence_weighting_old, write_weighting):
        """
        Updating the link matrix takes some effort (in order to vectorize the implementation)
        Instead of the original index-by-index operation, it's all done at once.
        
        
        :param link_matrix_old: from previous time step, shape [batch_size, memory_size, memory_size]
        :param precedence_weighting_old: from previous time step, shape [batch_size, memory_size]
        :param write_weighting: from current time step, shape [batch_size, memory_size]
        :return: updated link matrix
        """

        # --------------------------------------------------------------------------------------------------------------
        #link_matrix_old = tf.Print(link_matrix_old, [link_matrix_old[0, 0, 0], link_matrix_old[0, 0, 1], link_matrix_old[0, 1, 0], link_matrix_old[0, 1, 1]],
        #"\nlink_matrix_old_in_update_link_matrix_function: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        #precedence_weighting_old = tf.Print(precedence_weighting_old, [precedence_weighting_old[0, 0], precedence_weighting_old[0, 1]],
        #"\nprecedence_weighting_old_in_update_link_matrix_function: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        #write_weighting = tf.Print(write_weighting, [write_weighting[0, 0], write_weighting[0, 1]],
        #"\nwrite_weighting_in_update_link_matrix_function: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        expanded = tf.expand_dims(write_weighting, axis = 2)

        # vectorizing the paper's original implementation
        w = tf.tile(expanded, [1, 1, self.memory_size])  # shape [batch_size, memory_size, memory_size]

        # --------------------------------------------------------------------------------------------------------------
        #w = tf.Print(w, [w[0, 0, 0], w[0, 0, 1]], "\nw_in_update_link_matrix_function: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------


        # shape of w_transpose is the same: [batch_size, memory_size, memory_size]
        w_transp = tf.tile(tf.transpose(expanded, [0, 2, 1]), [1, self.memory_size, 1])

        # ------------------------------------- -------------------------------------------------------------------------
        #w_transp = tf.Print(w_transp, [w_transp[0, 0, 0], w_transp[0, 0, 1], w_transp[0, 1, 0], w_transp[0, 1, 1]], 
        #"\nw_transp_in_update_link_matrix_function: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------


        # in einsum, m and n are the same dimension because tensorflow doesn't support duplicated subscripts. Why?
        lm = (1 - w - w_transp) * link_matrix_old + tf.einsum("bn,bm->bmn", precedence_weighting_old, write_weighting)

        # --------------------------------------------------------------------------------------------------------------
        #lm = tf.Print(lm, [lm[0, 0, 0], lm[0, 0, 1], lm[0, 1, 0], lm[0, 1, 1]], 
        #"\nlm: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        lm *= (1 - tf.eye(self.memory_size, batch_shape = [self.batch_size], dtype = tf.float64))  # making sure self links are off

        # --------------------------------------------------------------------------------------------------------------
        #lm = tf.Print(lm, [lm[0, 0, 0], lm[0, 0, 1], lm[0, 1, 0], lm[0, 1, 1]], 
        #"\nlm: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        return tf.identity(lm, name = "Link_matrix")

    @staticmethod
    def content_based_addressing(memory, keys, strength):
        """
        :param keys: array of shape [batch_size, n_keys, word_size], i.e. [b, r, w]
        :param memory: array of shape [batch_size, memory_size, word_size], i.e. [b, n, w]
        :param strength: array of shape [batch_size, n_keys], i.e. [b, r]
        :return: tensor of shape [batch_size, memory_size, n_keys]
        """

        keys = tf.nn.l2_normalize(keys, dim = 2)

        # --------------------------------------------------------------------------------------------------------------
        #keys = tf.Print(keys, [keys[0, 0, 0]], "\nkeys_in_content_based_addressing_func: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        #strength = tf.Print(strength, [strength[0, 0]], "\nstrength_in_content_based_addressing_func: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        memory = tf.nn.l2_normalize(memory, dim = 2)

        # --------------------------------------------------------------------------------------------------------------
        #memory = tf.Print(memory, [memory[0, 0, 0], memory[0, 1, 0]], "\nMemory_in_content_based_addressing_func: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        similarity = tf.einsum("bnw,brw,br->bnr", memory, keys, strength)

        # --------------------------------------------------------------------------------------------------------------
        #similarity = tf.Print(similarity, [similarity[0, 0], similarity[0, 1]], "\nSimilarity_in_content_based_addressing_func: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        content_weighting = tf.nn.softmax(similarity, dim = 1, name = "Content_weighting")

        # --------------------------------------------------------------------------------------------------------------
        #content_weighting = tf.Print(content_weighting, [content_weighting[0, 0], content_weighting[0, 1]], "\nContent weighting_in_content_based_addressing_func: \n", summarize = 1)
        # --------------------------------------------------------------------------------------------------------------

        return content_weighting

    @staticmethod
    def oneplus(x):
        return 1 + tf.nn.softplus(x)

    def reshape_and_softmax(self, r_read_modes):
        r_read_modes = tf.reshape(r_read_modes, [self.batch_size, self.num_read_heads, 3])
        return tf.nn.softmax(r_read_modes, dim = 2)
