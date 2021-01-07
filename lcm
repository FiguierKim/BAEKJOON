import sys
input = sys.stdin.readline

def gcd(a, b):
    return gcd(b, a % b) if b else a
def lcm(a, b):
    return a * b // gcd(a, b)

t = int(input())
for i in range(t):
    a, b = map(int, input().split())
    print(lcm(a, b))
