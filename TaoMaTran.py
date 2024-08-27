import numpy as np 

# Tao ma tran D
D = np.array([ 
[3, 0, 0, 0], 
[0, 1, 0, 0], 
[0, 0, 2, 0], 
[0, 0, 0, 2]])
print("Man tran D:\n")
print(D)

# Ma tran nghich dao cua D
DNghichDao = np.linalg.inv(D) 
print("\nMa tran nghich dao cua D:\n")
print(DNghichDao)

# Ma tran nghich dao tang cuong của D
DNghichDao2 = np.linalg.inv(D + np.identity(4)) 
print("\nMa tran nghich dao tang cuong cua D:\n")
print(DNghichDao2)
