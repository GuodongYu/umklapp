from umklapp.umklapp_incommensurate import subspace, IncommenBilayer
from tBG.brillouin_zones import BZHexagonal
import numpy as np

icb = IncommenBilayer()
Qs_bott = subspace(icb.lattice_bott.latt_vec, 4.065040650406504)
Qs_top = subspace(icb.lattice_top.latt_vec, 4.065040650406504)
bz_bott = BZHexagonal(icb.lattice_bott.latt_vec)
bz_top = BZHexagonal(icb.lattice_top.latt_vec)

def plot_BZ(ax, bz, color):
    Ks = bz.special_points()['K']
    Ks_ = np.append(Ks, [Ks[0]], axis=0)
    ax.plot(Ks_[:,0], Ks_[:,1], color=color)
    for K in Ks:
        ax.plot([K[0],2*K[0]],[K[1], 2*K[1]], color=color)

def plot_Qs(ax, Qs, color):
    for i in range(len(Qs)):
        Q = Qs[i]
        ax.scatter(Q[0], Q[1], color=color)
        ax.text(Q[0], Q[1], '$Q_{%s}$' % i,color=color)

from matplotlib import pyplot as plt
fig, ax = plt.subplots(1,1, figsize=[10, 10])
plot_BZ(ax, bz_bott, 'blue')
plot_BZ(ax, bz_top, 'red')
plot_Qs(ax, Qs_bott, 'blue')
plot_Qs(ax, Qs_top, 'red')

ax.axis('equal')
plt.show()

    
