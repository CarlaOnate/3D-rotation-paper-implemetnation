# 3.1.2 Determination of the Spheroid Principal Axes

from statistics import mean
import math

# Spherical model:

a_i = [2,5,1,9]
b_i = [4,9,5,10]

res = [(x + y) / 2 for x, y in zip(a_i, b_i)]
print(res)

def Avg(res):
    return sum(res) / len(res)
# A_spherical = Avg(res)
A_spherical = mean(res)
print('A_spherical = B_spherical = ', A_spherical)

# Oblate model:

a_i = [2,5,1,9]
b_i = [4,9,5,10]

A_oblate = mean(a_i)
print('A_oblate = ', A_oblate)
B_oblate = min(b_i)
print('B_oblate = ', B_oblate)

# Prolate model:

a_i = [2,5,1,9]
b_i = [4,9,5,10]

A_prolate = max(a_i)
print('A_prolate = ', A_prolate)
B_prolate = mean(b_i)
print('B_prolate = ', B_prolate)

# 3.1.3 Elevation Angle Estimation
# b???
b = 5
cos_theta = math.sqrt((pow(b, 2) - pow(B_oblate, 2))/(pow(A_oblate, 2) - pow(B_oblate, 2)))
print('cos_theta = ',cos_theta)