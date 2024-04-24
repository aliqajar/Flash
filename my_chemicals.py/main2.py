


import numpy as np
import chemicals
from chemicals.vectorized import Antoine


# # Your array of temperatures
# temperatures = np.linspace(100, 200, 5)

# # Using the vectorized Antoine function directly
# pressures = Antoine(temperatures, A=8.95894, B=510.595, C=-15.95)

# print(pressures)

# result = chemicals.rachford_rice.Rachford_Rice_solution_binary_dd(zs=[1E-27, 1.0], Ks=[1000000000000,0.1])
# print(result)


# zs = [0.004632150100959984, 0.019748784459594933, 0.0037494212674659875, 0.0050492815033649835, 7.049818284201636e-05, 0.019252941309184937, 0.022923068733233923, 0.02751809363371991, 0.044055273670258854, 0.026348159124199914, 0.029384949788372902, 0.022368938441593926, 0.03876345111451487, 0.03440715821883388, 0.04220510198067186, 0.04109191458414686, 0.031180945124537895, 0.024703227642798916, 0.010618543295340965, 0.043262442161003854, 0.006774922650311977, 0.02418090788262392, 0.033168278052077886, 0.03325881573680989, 0.027794535589044905, 0.00302091746847699, 0.013693571363003955, 0.043274465132840854, 0.02431371852108292, 0.004119055065872986, 0.03314056562191489, 0.03926511182895087, 0.0305068048046159, 0.014495317922126952, 0.03603737707409988, 0.04346278949361786, 0.019715052322446934, 0.028565255195219907, 0.023343683279902924, 0.026532427286078915, 2.0833722372767433e-06]
# Ks = [0.000312001984979, 0.478348350355814, 0.057460349529956, 0.142866526725442, 0.186076915390803, 1.67832923245552, 0.010784509466239, 0.037204384948088, 0.005359146955631, 2.41896552551221, 0.020514598049597, 0.104545054017411, 2.37825397780443, 0.176463709057649, 0.000474240879865, 0.004738042026669, 0.02556030236928, 0.00300089652604, 0.010614774675069, 1.75142303167203, 1.47213647779132, 0.035773024794854, 4.15016401471676, 0.024475125100923, 0.00206952065986, 2.09173484409107, 0.06290795470216, 0.001537212006245, 1.16935817509767, 0.001830422812888, 0.058398776367331, 0.516860928072656, 1.03039372722559, 0.460775800103578, 0.10980302936483, 0.009883724220094, 0.021938589630783, 0.983011657214417, 0.01978995396409, 0.204144939961852, 14.0521979447538]

num_components = 10_000

zs = np.random.rand(num_components)
zs /= np.sum(zs)

beta = np.random.normal(loc=-4, scale=10.0, size=(num_components,))
Ks = np.exp(beta)



LF, VF, xs, ys = chemicals.rachford_rice.Rachford_Rice_solution_mpmath(zs=zs, Ks=Ks)
(LF*xs[-1] + VF*ys[-1])/zs[-1]

print(xs)
print(ys)


xs = np.array(xs)
ys = np.array(ys)

# Print negative liquid mole fractions
negative_x = xs[xs < 0]
if negative_x.size > 0:
    print("Negative liquid mole fractions (x):")
    print(negative_x)
else:
    print("No negative liquid mole fractions found.")

# Print negative vapor mole fractions
negative_y = ys[ys < 0]
if negative_y.size > 0:
    print("Negative vapor mole fractions (y):")
    print(negative_y)
else:
    print("No negative vapor mole fractions found.")

print(LF, VF)
print((LF*xs[-1] + VF*ys[-1])/zs[-1])

