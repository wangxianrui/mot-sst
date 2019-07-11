import numpy as np

data = np.random.rand(3, 4)
np.savetxt('test', data, fmt='%.4f', delimiter=',')
