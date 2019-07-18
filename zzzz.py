import numpy as np

data = np.random.rand(5, 1)
a = np.repeat(data,3,axis=1)
print(a.shape)
