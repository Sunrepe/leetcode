def solution_from():
    pass

def solution_mine():
    pass

if __name__ == '__main__':
    # solution_from()
    n, test = input().split()
    n, edges = int(n), int(test)
    gra = {}
    cities = set()
    for _ in range(edges):
        s, e, dist = input().split()
        if s not in cities:
            cities.add(s)
        if e not in cities:
            cities.add(e)
        gra.update({f"{s}-{e}": int(dist)})
        gra.update({f"{e}-{s}": int(dist)})
    test = int(input())
    def check_travel(travel, path):
        flag = 1 # simple cycle
        vis = {travel[0]}
        tot_dis = 0
        n = len(travel)
        for i in range(n-1):
            dis = gra.get(f"{travel[i]}-{travel[i+1]}")
            if dis:
                tot_dis += dis
                if i < n-2 and travel[i+1] in vis:
                    flag = 2
                if i==n-2 and travel[i+1]!=travel[0]:
                    flag=3
                vis.add(travel[i+1])
            else:
                return [path, None, 3]
        if vis != cities:
            flag=3
        return [path, tot_dis, flag]

    minx = 90000
    res = [None, None]
    for path in range(1, test+1):
        travel = input().split()[1:]
        tmp_path, tot, flag = check_travel(travel, path)
        if not tot:
            print(f"Path {tmp_path}: NA (Not a TS cycle)")
        elif flag==3:
            print(f"Path {tmp_path}: {tot} (Not a TS cycle)")
        elif flag==1:
            print(f"Path {tmp_path}: {tot} (TS simple cycle)")
            if tot<minx:
                minx = tot
                res = [path, tot]
        else:
            print(f"Path {tmp_path}: {tot} (TS cycle)")
            if tot<minx:
                minx = tot
                res = [path, tot]
    print(f"Shortest Dist({res[0]}) = {res[1]}")
