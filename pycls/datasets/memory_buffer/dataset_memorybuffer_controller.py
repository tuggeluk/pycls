import os
from pycls.datasets.memory_buffer.dataset_memorybuffer import DatasetMemoryBuffer
import time
"""Loads a complete dataset into memory, in order do reduce disk load during training"""


def data_memory_loop(cont_queue, name):

    not_done = True
    dsm = DatasetMemoryBuffer(name)

    def load(name, data_dir, usepickle = False):
        if usepickle:
            load_file = os.path.join(data_dir+"_cache", "train_"+name+".pickle")
        else:
            load_file = os.path.join(data_dir, name, "train")
        dsm.load(load_file)

    def unlink():
        dsm.unlink()


    while not_done:
        queue_result = cont_queue.get()

        if queue_result[0] == "done":
            unlink()
            not_done = False
            cont_queue.put("done")
            time.sleep(3)
        elif queue_result[0] == "load":
            load(queue_result[1], queue_result[2])
            cont_queue.put("loaded")
            time.sleep(3)
        elif queue_result[0] == "empty":
            unlink()
            cont_queue.put("emptied")
            time.sleep(3)
        else:
            print("unknown command: " + queue_result)





