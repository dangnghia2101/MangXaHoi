import numpy as np 
D = np.array([ 
[3, 0, 0, 0], 
[0, 1, 0, 0], 
[0, 0, 2, 0], 
[0, 0, 0, 2]])

A = np.array([
 [1, 1, 1, 1],
 [1, 1, 0, 0],
 [1, 0, 1, 1],
 [1, 0, 1, 1]
])

print("Cong thuc 1: Chuan hoa moi hang cua ma tran")
print(np.linalg.inv(D + np.identity(4)) @ A)

print("\nCong thuc 2: Chuan hoa moi cot cua ma tran")
print(A @ np.linalg.inv(D + np.identity(4)))
