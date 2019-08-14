'''
@Author: rayenwang
@Date: 2019-08-14 17:37:16
@Description: 
'''

a = '100'
b = '1'
max_len = max(len(a), len(b))
a = [int(n) for n in list(a.zfill(max_len))]
b = [int(n) for n in b.zfill(max_len)]
c = [0] * max_len
flag = 0
for i in range(max_len)[::-1]:
    num = a[i] + flag - b[i]
    c[i] = num % 10
    flag = num // 10
print(c)
