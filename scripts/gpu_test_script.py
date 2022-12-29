USE_CUDA=True

if USE_CUDA:
    import cupy as npcp
    DeviceContext = npcp.cuda.Device
else:
    import numpy as npcp
    npcp.asnumpy = lambda x: x
    import contextlib
    DeviceContext = contextlib.nullcontext

import multiprocessing
from multiprocessing import Process, Queue
import time
import os

N = 2000

def f(x):
    #get some data from the global context
    global glob_var

    #get the device ID from the global context
    global device_id
    
    print("f: process PID {}, GPU {}".format(multiprocessing.current_process().pid, device_id))

    #ensure computation is on the device allocated to this process
    with DeviceContext(device_id):
        y = x*npcp.ones((glob_var.shape[0], glob_var.shape[1], glob_var.shape[1]), dtype=npcp.float32)
        ret = y @ glob_var
        ret = ret.swapaxes(1,2) @ y
        ret = npcp.asnumpy(ret.mean())
    return ret

def init_vars(queue):
    global glob_var
    global device_id

    #get the device assigned to this process
    device_id = queue.get()
    # print("init_vars: process PID {}, GPU {}".format(multiprocessing.current_process().pid, device_id))

    #allocate data on the device
    with DeviceContext(device_id):
        glob_var = npcp.ones((N, 6, 1), dtype=npcp.float32)

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
  
    print("Started main..")
    if USE_CUDA: 
        cvd = os.environ["CUDA_VISIBLE_DEVICES"]
        cvd = [int(x) for x in cvd.split(",")]
        NUM_GPUS = len(cvd)
        print("Num gpus {}".format(NUM_GPUS))
        print("Num cpus {}".format(multiprocessing.cpu_count()))
    #actually no GPUs will be used, we just create 1xPROC_PER_GPU CPU processes
    else:
        NUM_GPUS = 1
 
    PROC_PER_GPU = 8
    queue = Queue()
    #even though CUDA_VISIBLE_DEVICES could be e.g. 3,4
    #here the indexing will be from 0,1, as nvidia hides the other devices
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    t0 = time.time()
    input_data = range(1000)

    pool = multiprocessing.Pool(NUM_GPUS*PROC_PER_GPU, initializer=init_vars, initargs=(queue, ))

    #calculate the output 3 times
    for i in range(3):
        ret = sum(pool.map(f, input_data))
        print(ret)

    #ensure processes are closed
    pool.close()
    pool.join()

    #get the time
    t1 = time.time()
    print(t1 - t0)


