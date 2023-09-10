from multiprocessing import Process
import os


# 子进程要执行的代码
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))
    for i in range(100000000):
        pass


if __name__ == '__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))

    print('Child process will start.')
    p.start()
    p.join()

    print('Child process end.')


# Parent process 928.
# Child process will start.
# Run child process test (929)...
# Process end.