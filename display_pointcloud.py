import pptk
import numpy as np
p = np.load("/tmp/pc.npy").squeeze()
print(p)
print(p.shape)
v = pptk.viewer(p)