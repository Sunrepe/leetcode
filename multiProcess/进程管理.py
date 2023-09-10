import multiprocessing, os


def mod(d, l):
    d[os.getpid()] = os.getppid()
    l.append(os.getpid())
    print("\033[1;32m%s\033[0m" % l)
    print("\033[1;35m%s\033[0m" % d)


if __name__ == "__main__":
    M = multiprocessing.Manager()
    # with multiprocessing.Manager() as M:    与上面的M = multiprocessing.Manager() 作用一致
    dic = M.dict()  # 生成一个字典，可在多个进程间共享和传递
    list1 = M.list(range(2))  # 生成一个列表，可在多个进程间共享和传递
    p_list = []
    for i in range(5):
        p = multiprocessing.Process(target=mod, args=(dic, list1))
        p.start()
        p_list.append(p)
    for res in p_list:
        res.join()  # 等待结果，不然程序会报错

    print(dic)
    print(list1)