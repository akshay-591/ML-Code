import numpy as mat
from MLTricks import stand_dev, feature_normalization
a = mat.array([[1,2],
              [2, 5]])
a = feature_normalization.normalize(a)
a= a[0]
print(a)
print(mat.min(a))
