import sys
def dis(a,b):
    return (b[0]-a[0])**2 + (b[1]-a[1]) ** 2
def devi(st, en):
    leng = en - st
    if leng == 2:
        return dis(s[st], s[st+1])
    elif leng == 3:
        return min(dis(s[st], s[st+1]), dis(s[st+1], s[st+2]), dis(s[st], s[st+2]))
    
    leng = (st + en) // 2
    mid = s[leng][0]
    d = min(devi(st,leng), devi(leng, en))
    tmp = []
    for i in range(st,en):
        if (mid - s[i][0])**2 <= d:
            tmp.append(s[i])
    tmp.sort(key = lambda x:x[1])
    m = d
    tmp_len = len(tmp)
    if tmp_len >= 2:
        for i in range(tmp_len -1):
            for j in range(i+1, tmp_len):
                if (tmp[i][1] - tmp[j][1]) **2 > d:
                    break
                elif tmp[i][0] < mid and tmp[j][0] < mid:
                    continue
                elif tmp[i][0] > mid and tmp[j][0] > mid:
                    continue
                m = min(m, dis(tmp[i],tmp[j]))
    return m
    
li = []
for _ in range(int(input())):
    li.append(list(map(int, sys.stdin.readline().split())))
s = list(set(map(tuple,li)))
if len(s) != len(li):
    print("0")
else:
    s.sort()
    d = devi(0,len(s))
    print(d)
