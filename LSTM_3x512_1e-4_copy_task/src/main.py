# -*- coding: utf-8 -*-

from controller_implementations.dnc.dnc import *
from controller_implementations.feedforward import *
from controller_implementations.lstm import *

from task_implementations.copy_task import *
from utils import project_path

n_blocks = 7
vector_size = n_blocks + 1   # input and ouput vector sizes
min_seq = 1
train_max_seq = 20
n_copies = 1  
out_vector_size = vector_size  # input and ouput vector sizes

# ------------------------------------------------------------------------------------
task_for_training = CopyTask(vector_size, min_seq, train_max_seq, n_copies)
Number_of_test_obj_by_each_length_of_binary_vectors_secuence = 100
# ------------------------------------------------------------------------------------

#task = CopyTask(vector_size, min_seq, train_max_seq, n_copies)
#task = bAbITask(os.path.join("tasks_1-20_v1-2", "en-10k"))

print("Loaded task")

class Hp:
    """
    Hyperparameters
    """
    batch_size = 4
    steps = 200000

    lstm_memory_size = 256

    n_layers = 3

    stddev = 0.1

    class Mem:
        word_size = 15
        mem_size = 25
        num_read_heads = 4

#controller = Feedforward(task.vector_size, Hp.batch_size, [3, 3])

# out_vector_size = task.vector_size is a needed argument to LSTM if you want to test just the LSTM outside of DNC
controller = LSTM(inp_vector_size = task_for_training.vector_size, memory_size = Hp.lstm_memory_size, n_layers = Hp.n_layers, batch_size = Hp.batch_size, out_vector_size = out_vector_size, initial_stddev = Hp.stddev)

#dnc = DNC(controller = controller, batch_size = Hp.batch_size, out_vector_size = task_for_training.vector_size, mem_hp = Hp.Mem, initial_stddev = Hp.stddev)

print("Loaded controller")

#restore_path = os.path.join(project_path.log_dir, "June_09__19:32", "train", "model.chpt")

controller.run_session(task_for_training, vector_size, n_copies, Number_of_test_obj_by_each_length_of_binary_vectors_secuence, Hp, project_path)  # , restore_path = restore_path)