import numpy as mat

a = mat.array([[1, 2, 3], [2, 4, 5]])
c = a*5

print(a)
print("\n\n")
r = mat.r_[a.flatten(),c.flatten()]
print(r)
b = mat.reshape(r[0:6], (2, 3))
print(b)
d = mat.reshape(r[6:r.shape[0]],(2,3))
print(d)