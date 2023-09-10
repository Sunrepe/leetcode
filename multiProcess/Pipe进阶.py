from multiprocessing import Pipe, Process
import time
import os

def son_process(x, pipe: Pipe()):
    print("子进程：{}".format(os.getpid()))
    _out_pipe, _in_pipe = pipe

    # 关闭fork过来的输入端
    _out_pipe.close()
    for i in range(5):
        time.sleep(1)
        _in_pipe.send('x_:'+str(i))
    print("子进程 end")


if __name__ == '__main__':
    print('Main process: {}.'.format(os.getpid()))
    out_pipe, in_pipe = Pipe(True)
    son_p = Process(target=son_process, args=(5, (out_pipe, in_pipe)))
    son_p.start()

    # 等 pipe 被 fork 后，关闭主进程的输出端
    # 这样，创建的Pipe一端连接着主进程的输入，一端连接着子进程的输出口
    in_pipe.close()
    while True:
        try:
            sms = out_pipe.recv()
            print(sms)
        except EOFError:
            break
    son_p.join()
    out_pipe.close()
    print("主进程也end了")