
import numpy as mat

a=[1,2,3,4]
a_mat = mat.array(a)
print(a_mat.ndim)
a_diag = mat.diag(a_mat)
mat.linalg.pinv
print(a_diag)