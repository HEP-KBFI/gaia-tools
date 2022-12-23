import cupy
import multiprocessing
from multiprocessing import Process
import time

def f(x):
    print("proc", multiprocessing.current_process().pid, x)
    #global glob_var
    #global mingi_var

    mingi_var = 'mingi teine tekst'
    print(mingi_var)


    y = x*cupy.ones(glob_var.shape[0])
    ret = glob_var + y
    ret = cupy.asnumpy(ret)
    time.sleep(2)
    return ret

def init_vars(arg1):
    global glob_var
    global mingi_var

    mingi_var = arg1
    glob_var = cupy.zeros(10)
    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    input_data = range(10)

    arg1 = 'mingi tekst'

    pool = multiprocessing.Pool(4, initializer=init_vars, initargs=(arg1,))
    ret = pool.map(f, input_data)

    print('Pool done')
    for r in ret:
        print(r)


    pool.close()
    pool.join()