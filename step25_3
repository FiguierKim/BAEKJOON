import heapq
import sys
heap = []
for _ in range(int(input())):
    num = int(sys.stdin.readline())
    if num != 0:
        heapq.heappush(heap,(abs(num),num))
    else:
        if heap:
            print(heapq.heappop(heap)[1])
        else:
            print(0)
