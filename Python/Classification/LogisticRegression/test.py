
import numpy as mat

a = [[1, 2],
     [4, 3],
     [4, 5]]



b = mat.c_[a]
c= b[:, 0]
print(c)

d=c[mat.ix_([0, 1, 2])]
print(d)

index = mat.where(d == 4)
print(index[0])

e=d[mat.ix_(index[0])]

print(e)

