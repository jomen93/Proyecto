import numba as nb
from collections import OrderedDict

spec = OrderedDict()
spec["x"] = nb.float32
spec["y"] = nb.float32
spec["z"] = nb.float32

@nb.jitclass(spec)
class Vec(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.z = 3

	def add(self,dx,dy):
		self.x += dx
		self.y += dy


Xvec = Vec(1,1)
print(Xvec.x)		
print(Xvec.y)		
Xvec.add(0.5,0.5)
print(Xvec.x)		
print(Xvec.y)		
print(Xvec.z)		