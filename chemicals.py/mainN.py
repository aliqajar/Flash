

import numpy as np
import chemicals


num_components = 1_000

zs = np.random.rand(num_components)
zs /= np.sum(zs)

beta = np.random.normal(loc=0, scale=1.0, size=(num_components,))
Ks_12 = np.exp(beta)

beta = np.random.normal(loc=-0.3, scale=2.0, size=(num_components,))
Ks_13 = np.exp(beta)

beta = np.random.normal(loc=-1, scale=1.5, size=(num_components,))
Ks_14 = np.exp(beta)

beta = np.random.normal(loc=1.3, scale=4.0, size=(num_components,))
Ks_15 = np.exp(beta)

beta = np.random.normal(loc=-0.2, scale=7, size=(num_components,))
Ks_16 = np.exp(beta)

beta = np.random.normal(loc=0.5, scale=2.5, size=(num_components,))
Ks_17 = np.exp(beta)

beta = np.random.normal(loc=-3.2, scale=3.0, size=(num_components,))
Ks_18 = np.exp(beta)

beta = np.random.normal(loc=0.2, scale=1.1, size=(num_components,))
Ks_19 = np.exp(beta)

beta = np.random.normal(loc=-0.3, scale=2.0, size=(num_components,))
Ks_110 = np.exp(beta)

beta = np.random.normal(loc=-1, scale=5, size=(num_components,))
Ks_111 = np.exp(beta)

beta = np.random.normal(loc=-3.6, scale=4.0, size=(num_components,))
Ks_112 = np.exp(beta)

beta = np.random.normal(loc=-2.2, scale=7, size=(num_components,))
Ks_113 = np.exp(beta)

beta = np.random.normal(loc=1.5, scale=2.1, size=(num_components,))
Ks_114 = np.exp(beta)

beta = np.random.normal(loc=0.2, scale=0.3, size=(num_components,))
Ks_115 = np.exp(beta)

beta = np.random.normal(loc=-1.2, scale=3.0, size=(num_components,))
Ks_116 = np.exp(beta)

phase_fraction, xr = chemicals.rachford_rice.Rachford_Rice_solutionN(
    zs, 
    [Ks_12, Ks_13, Ks_14, Ks_15, Ks_16, Ks_17, Ks_18, Ks_19, Ks_110, Ks_111, Ks_112, Ks_113, Ks_114, Ks_115, Ks_116], 
    [.1, .03, 0.05, 0.3, 0.1, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.1, 0.01, 0.01, 0.03]
    )

print(phase_fraction)

# x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = xr

# x1 = np.array(x1)
# x2 = np.array(x2)
# x3 = np.array(x3)
# x4 = np.array(x4)
# x5 = np.array(x5)
# x6 = np.array(x6)
# x7 = np.array(x7)
# x8 = np.array(x8)
# x9 = np.array(x1)
# x10 = np.array(x2)
# x11 = np.array(x3)
# x12 = np.array(x4)
# x13 = np.array(x5)
# x14 = np.array(x6)
# x15 = np.array(x7)
# x16 = np.array(x8)

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

# negative_x = x4[x4 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x4):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x5[x5 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x5):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x6[x6 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x6):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x7[x7 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x7):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x8[x8 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x8):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x9[x9 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x9):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x10[x10 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x10):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x11[x11 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x11):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x12[x12 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x12):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x13[x13 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x13):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")        

# negative_x = x14[x14 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x14):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x15[x15 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x15):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")

# negative_x = x16[x16 < 0]
# if negative_x.size > 0:
#     print("Negative mole fractions (x16):")
#     print(negative_x)
# else:
#     print("No negative mole fractions found.")




