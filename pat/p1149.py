
if __name__ == '__main__':

    n, test = input().split()
    n, test = int(n), int(test)


    d = {}
    l = []

    for i in range(n):
        a, b = input().split()
        # a, b = int(a), int(b)
        if d.get(a):
            d[a].append(b)
        else:
            d[a] = [b]
        if d.get(b):
            d[b].append(a)
        else:
            d[b] = [a]

    for i in range(test):
        res = input().split()
        m = int(res[0])
        flag = False
        s = set()
        for j in range(m):
            id = res[j+1]
            if id in s:
                flag = True
                break
            elif d.get(id):
                s.update(d[id])

        if flag:
            print('No')
        else:
            print('Yes')