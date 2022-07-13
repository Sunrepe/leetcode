from multiprocessing import Process, Queue
import os, time, random

def go_abs(p):
    for i in range(5):
        p.put(i)
        time.sleep(1)

# 写数据进程执行的代码:
def write(q):
    print('Process to write: %s' % os.getpid())
    time.sleep(1)
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())
    go_abs(q)

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    # time.sleep(3)
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    q.put('Hello')
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    pw.start()  # 启动子进程pw，写入:
    pr.start()  # 启动子进程pr，读取:
    pw.join()   # 等待pw结束

    pr.terminate()  # pr进程里是死循环，无法等待其结束，只能强行终止:
