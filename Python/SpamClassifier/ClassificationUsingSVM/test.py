import numpy as mat
a = [1, 2, 9, 5, 6, 8]
d = mat.c_[mat.array(a)]
print(mat.sort(d,axis=0)[::-1])