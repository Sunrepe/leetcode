from multiprocessing import Pool
import os, time, random

def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(6):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

#
# $ nslookup www.python.org
# Server:		192.168.19.4
# Address:	192.168.19.4#53
#
# Non-authoritative answer:
# www.python.org	canonical name = python.map.fastly.net.
# Name:	python.map.fastly.net
# Address: 199.27.79.223
#
# Exit code: 0