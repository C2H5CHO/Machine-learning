import numpy as np
import pandas as pd

x = np.array([[1, 1], [3, 1]])
print(f"x: {x}")
y = np.array([2, 4]).reshape(2, 1)
print(f"y: {y}")

print(f"w, b: {np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)}")
print(f"w, b: {np.linalg.lstsq(x, y, rcond=-1)}")