
if __name__ == '__main__':

    n = int(input())

    say = [0]

    for i in range(n):
        a = int(input())
        say.append(a)

    def check(sta, wa, wb):
        if sta <= 0:
            if sta == -wa or sta == -wb:
                return 0
            else: return 1
        if sta > 0:
            if sta == wa or sta == wb:
                return 1
            else: return 0


    find, ans = False, []
    for i in range(1, n+1):
        if find: break
        for j in range(i+1, n+1):
            # check wovle
            lys = 0
            lys += check(say[i], i, j)
            lys += check(say[j], i, j)
            if lys != 1:
                continue
            # check human
            lys = 0
            for s in say[1:]:
                lys += check(s, i, j)
            if lys == 2:
                find = True
                ans.append(i)
                ans.append(j)
    if find:
        print('{} {}'.format(ans[0], ans[1]))
    else:
        print('No Solution')

