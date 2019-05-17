# -*- coding: utf-8 -*-

import tensorflow as tf
from task_implementations.copy_task import *
from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug


class Controller:
    """
    Controller: a neural network that optionally uses memory or other controllers to produce output.
    Контроллер - это нейронная сеть, которая опционально использует память или другие контроллеры для генерации выходных данных.

    All neural networks should inherit from this class.
    Вне нейронные сети должны наследовать этот класс.

    FF network is a controller, a LSTM network is a controller, but DNC is also a controller (which uses another
    controller inside it)
    Сеть прямого распространения - это контроллер, LSTM сеть - тоже контроллер, но DNC - это контроллер (который использует другой контроллер внтутри себя)

    The only problem is that this is theory, and the way tf.while_loop is implemented prevents easy nesting of 
    controllers :/
    Единственная проблема заключается в том, что это теория, и реализованый способ tf.while_loop предотвращает легкую (простую) вложенность контроллеров.
    """
    
    max_outputs = 1
    clip_value = 10

    def run_session(self,
                    task_for_training,
                    vector_size, n_copies,
                    Number_of_test_obj_by_each_length_of_binary_vectors_secuence,
                    hp,
                    project_path,
                    restore_path = None,
                    optimizer = tf.train.RMSPropOptimizer(learning_rate = 1e-4, momentum = 0.9)):

        # -----------------------------------------------------------------------------------------------------------

        x = tf.placeholder(tf.float64, task_for_training.x_shape, name = "X")
        y = tf.placeholder(tf.float64, task_for_training.y_shape, name = "Y")

        sequence_lengths = tf.placeholder(tf.int64, [None], name = "Sequence_length_train")
        mask = tf.placeholder(tf.float64, task_for_training.mask, name = "Output_mask_train")

        outputs, summaries = self(x, sequence_lengths)
        assert tf.shape(outputs).shape == tf.shape(y).shape

        # TODO this always needs to be changed depending on the task because of the way tf implements losses
        summary_outputs = tf.nn.sigmoid(outputs)
        # summary_outputs = tf.nn.softmax(outputs, dim = 1)

        tf.summary.image("0_Input_train", tf.expand_dims(tf.transpose(x, [0, 2, 1]), axis = 3),
                         max_outputs = Controller.max_outputs)
        tf.summary.image("0_Network_output_train", tf.expand_dims(tf.transpose(summary_outputs, [0, 2, 1]), axis = 3),
                         max_outputs = Controller.max_outputs)
        tf.summary.image("0_Y_train", tf.expand_dims(tf.transpose(y * mask, [0, 2, 1]), axis = 3),
                         max_outputs = Controller.max_outputs)

        cost = task_for_training.cost(outputs, y, mask)

        tf.summary.scalar("Cost", cost)

        # optimizer, clipping gradients and summarizing gradient histograms
        gradients = optimizer.compute_gradients(cost)

        from tensorflow.python.framework import ops

        for i, (gradient, variable) in enumerate(gradients):
            if gradient is not None:
                cur_variable_tensor_value_for_visualize = variable
                
                """
                # Здесь рассмотрены все варианты размерности тензора, которые могут быть в данной реализации DNC
                if len(cur_variable_tensor_value_for_visualize.get_shape()) == 1:
                    for index_1 in range(cur_variable_tensor_value_for_visualize.get_shape()[0]):
                        tf.summary.scalar(variable.name + "#_#" + str(index_1), cur_variable_tensor_value_for_visualize[index_1])

                if len(cur_variable_tensor_value_for_visualize.get_shape()) == 2:
                    for index_1 in range(cur_variable_tensor_value_for_visualize.get_shape()[0]):
                        for index_2 in range(cur_variable_tensor_value_for_visualize.get_shape()[1]):
                            tf.summary.scalar(variable.name + "#_#" + str(index_1) + '_' + str(index_2), cur_variable_tensor_value_for_visualize[index_1, index_2])
                    
                if len(cur_variable_tensor_value_for_visualize.get_shape()) == 3:
                    for index_1 in range(cur_variable_tensor_value_for_visualize.get_shape()[0]):
                        for index_2 in range(cur_variable_tensor_value_for_visualize.get_shape()[1]):
                            for index_3 in range(cur_variable_tensor_value_for_visualize.get_shape()[2]):
                                tf.summary.scalar(variable.name + "#_#" + str(index_1) + '_' + str(index_2) + '_' + str(index_3), cur_variable_tensor_value_for_visualize[index_1, index_2, index_3])
                """

                cur_variable_gradient = gradient.values if isinstance(gradient, ops.IndexedSlices) else gradient

                """
                # Здесь рассмотрены все варианты размерности тензора, которые могут быть в данной реализации DNC
                if len(cur_variable_gradient.get_shape()) == 1:
                    for index_1 in range(cur_variable_gradient.get_shape()[0]):
                        tf.summary.scalar(variable.name + "/gradients_before_cliping" + str(index_1), cur_variable_gradient[index_1])

                if len(cur_variable_gradient.get_shape()) == 2:
                    for index_1 in range(cur_variable_gradient.get_shape()[0]):
                        for index_2 in range(cur_variable_gradient.get_shape()[1]):
                            tf.summary.scalar(variable.name + "/gradients_before_cliping" + str(index_1) + '_' + str(index_2), cur_variable_gradient[index_1, index_2])
                    
                if len(cur_variable_gradient.get_shape()) == 3:
                    for index_1 in range(cur_variable_gradient.get_shape()[0]):
                        for index_2 in range(cur_variable_gradient.get_shape()[1]):
                            for index_3 in range(cur_variable_gradient.get_shape()[2]):
                                tf.summary.scalar(variable.name + "/gradients_before_cliping" + str(index_1) + '_' + str(index_2) + '_' + str(index_3), cur_variable_gradient[index_1, index_2, index_3])
                """

                clipped_gradient = tf.clip_by_value(gradient, -Controller.clip_value, Controller.clip_value)
                gradients[i] = clipped_gradient, variable

                #shape_len = len(tensor_value_for_visualize.get_shape())

                #tf.summary.scalar("shape_len: ", shape_len)

                """
                # Здесь рассмотрены все варианты размерности тензора, которые могут быть в данной реализации DNC
                if len(cur_variable_gradient.get_shape()) == 1:
                    for index_1 in range(cur_variable_gradient.get_shape()[0]):
                        tf.summary.scalar(variable.name + "/gradients_after_cliping" + str(index_1), cur_variable_gradient[index_1])

                if len(cur_variable_gradient.get_shape()) == 2:
                    for index_1 in range(cur_variable_gradient.get_shape()[0]):
                        for index_2 in range(cur_variable_gradient.get_shape()[1]):
                            tf.summary.scalar(variable.name + "/gradients_after_cliping" + str(index_1) + '_' + str(index_2), cur_variable_gradient[index_1, index_2])
                    
                if len(cur_variable_gradient.get_shape()) == 3:
                    for index_1 in range(cur_variable_gradient.get_shape()[0]):
                        for index_2 in range(cur_variable_gradient.get_shape()[1]):
                            for index_3 in range(cur_variable_gradient.get_shape()[2]):
                                tf.summary.scalar(variable.name + "/gradients_after_cliping" + str(index_1) + '_' + str(index_2) + '_' + str(index_3), cur_variable_gradient[index_1, index_2, index_3])
                """

        optimizer = optimizer.apply_gradients(gradients)

        self.notify(summaries)

        merged = tf.summary.merge_all()

        from numpy import prod, sum
        n_vars = sum([prod(var.shape) for var in tf.trainable_variables()])
        print("This model has", n_vars, "parameters!")

        saver = tf.train.Saver()

        sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        
        if restore_path is not None:
            saver.restore(sess, restore_path)
            print("Restored model", restore_path, "!!!!!!!!!!!!!")
        else:
            sess.run(tf.global_variables_initializer())


        train_writer = tf.summary.FileWriter(project_path.train_path, sess.graph)
        test_writer = tf.summary.FileWriter(project_path.test_path, sess.graph)
 
        from time import time
        t = time()

        cost_value = 9999
        print("Starting...")

        Q_log_file_path = 'Q_log_file.txt'
        Q_log_file = open(Q_log_file_path, 'w')

        for step in range(hp.steps):
            # Generates new curriculum training data based on current cost
            
            data_batch, seqlen, m, status_of_training = task_for_training.generate_data(cost = cost_value, batch_size = hp.batch_size, train = True)
            
            _, cost_value, summary = sess.run([optimizer, cost, merged], feed_dict = {x: data_batch[0], y: data_batch[1], sequence_lengths: seqlen, mask: m})
                                                
            if step % 100 == 0:
                print("Summary generated. Step", step, " Train cost == %.9f Time == %.2fs" % (cost_value, time() - t))
                t = time()
                #train_writer.add_summary(summary, step) # записываем лог всех необходимых данных модели

                Q_log_file.write("it " + str(step) + ": " + str(cost_value) + "\r\n")
                
                
            if step % 5000 == 0:
                saver.save(sess, project_path.model_path)
                print("Model saved!")

            
            if status_of_training == "Done with the training!":
                print("Training has been finished! :)")
                break


        saver.save(sess, project_path.model_path)
        print("Model saved!")

        train_writer.close()

        

        for test_length in range(1, 2):
            cost_value_sum = 0.0
            task_for_testing = CopyTask(vector_size, test_length, test_length, n_copies)
            for j in range(Number_of_test_obj_by_each_length_of_binary_vectors_secuence):
                data_batch, seqlen, m, _ = task_for_testing.generate_data(cost = cost, train = False, batch_size = hp.batch_size)
                summary, pred, cost_value = sess.run([merged, outputs, cost], feed_dict = {x: data_batch[0], y: data_batch[1], sequence_lengths: seqlen, mask: m})

                cost_value_sum += cost_value

                #test_writer.add_summary(summary, step)
                #task_for_testing.display_output(pred, data_batch, m)

                print("Summary generated. Step", step, " Test cost == %.9f Time == %.2fs" % (cost_value, time() - t))

                t = time()
            
            Q = cost_value_sum / Number_of_test_obj_by_each_length_of_binary_vectors_secuence
            Q_log_file.write("test_len: " + str(test_length) + "\r\n")
            Q_log_file.write(str(Q) + "\r\n")

        
        for test_length in range(10, 121, 10):
            cost_value_sum = 0.0
            task_for_testing = CopyTask(vector_size, test_length, test_length, n_copies)
            for j in range(Number_of_test_obj_by_each_length_of_binary_vectors_secuence):
                data_batch, seqlen, m, _ = task_for_testing.generate_data(cost = cost, train = False, batch_size = hp.batch_size)
                summary, pred, cost_value = sess.run([merged, outputs, cost], feed_dict = {x: data_batch[0], y: data_batch[1], sequence_lengths: seqlen, mask: m})

                cost_value_sum += cost_value

                #test_writer.add_summary(summary, step)
                #task_for_testing.display_output(pred, data_batch, m)

                print("Summary generated. Step", step, " Test cost == %.9f Time == %.2fs" % (cost_value, time() - t))

                t = time()

            Q = cost_value_sum / Number_of_test_obj_by_each_length_of_binary_vectors_secuence
            Q_log_file.write("test_length: " + str(test_length) + "\r\n")
            Q_log_file.write(str(Q) + "\r\n")
                
            
        Q_log_file.close()
        test_writer.close()
        sess.close()
            


    def notify(self, summaries):
        """
        Method which implements all the tf summary operations. If the instance uses another controller inside it, 
        like DNC, then it's responsible for calling the controller's notify method
        
        :param summaries: 
        :return: 
        """
        raise NotImplementedError()

    def __call__(self, x, sequence_length):
        """
        Returns all outputs after iteration for all time steps
        
        High level overview of __call__:
        for step in steps:
            output, state = self.step(data[step])
        self.notify(all_states)
        
        
        :param x: inputs for all time steps of shape [batch_size, input_size, sequence_length]
        :return: list of two tensors [all_outputs, all_summaries]
        """
        raise NotImplementedError()

    def step(self, x, state, step):
        """
        Returns the output vector for just one time step.
        But I'm not sure anymore how much does all of this work since because of the way tf.while_loop is implemented...
        
        :param x: one vector representing input for one time step
        :param state: state of the controller
        :param step: current time step
        :return: output of the controller and its current state
        """
        raise NotImplementedError()
