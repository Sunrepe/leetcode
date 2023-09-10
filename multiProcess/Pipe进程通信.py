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
    remotes, work_remotes = zip(*[Pipe() for _ in range(4)])

    ps = [Process(target=son_process, args=(12, (pp, sp))) for (pp, sp) in zip(work_remotes, remotes)]
    for p in ps:
        p.start()

    for r in work_remotes:
        r.close()

    print("主进程也end了")