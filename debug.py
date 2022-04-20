import numpy as np
from datetime import datetime

n = 5000*1000

a = datetime.now()
stuff = []
for i in range(n):
    stuff.append(i)

b = datetime.now()

print(b-a)

a = datetime.now()
stuff = np.zeros(n)
for i in range(n):
    stuff[i] = i


b = datetime.now()

print(b-a)