numbers = int(input())
num_list = list(map(int, input().split()))
stack = []
result = [-1 for _ in range(numbers)]

for i in range(len(num_list)):
    try:
        while num_list[stack[-1]] < num_list[i]:
            result[stack.pop()] = num_list[i]
    except:
        pass
        
    stack.append(i)
        
for i in range(numbers):
    print(result[i], end = ' ')
