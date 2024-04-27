



import numpy as np
import chemicals


num_components = 10

zs = np.random.rand(num_components)
zs = (zs-0.3) * 100
zs[0:2] = 0.1
print(zs)

beta = np.random.normal(loc=0, scale=1.0, size=(num_components,))
Ks_12 = np.exp(beta)

beta = np.random.normal(loc=-0.3, scale=0.4, size=(num_components,))
Ks_13 = np.exp(beta)


result = chemicals.rachford_rice.Rachford_Rice_solution2(zs, Ks_12, Ks_13, 0.1, 0.2)
L1, L2, x1, x2, x3 = result

# print(result)
print(L1, L2, 1-L1-L2, x1, x2, x3)

# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = xr

# x1 = np.array(x1)
# x2 = np.array(x2)
# x3 = np.array(x3)


# negative_x = x1[x1 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x1):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x2[x2 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x2):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x3[x3 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x3):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")


