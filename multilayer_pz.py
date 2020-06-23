import numpy as np
from tBG.hopping import filter_neig_list
from functools import reduce
import numpy.linalg as npla
from scipy.linalg.lapack import zheev
from scipy.linalg import eigh
from scipy.fftpack import fft2
from tBG.utils import *
from tBG.brillouin_zones import BZHexagonal, kpoints_line_mode
from tBG.hopping import hop_list_graphene_wannier, calc_hopping_pz_PBC,\
                  hop_params_wannier_interlayer,hop_func_wannier_interlayer, hop_list_graphene_wannier
import time
import matplotlib.collections as mc
from pymatgen.core.structure import Structure as pmg_struct
import copy


scale = 12
params = {'backend': 'ps',
          'axes.labelsize': 4*scale,
          'axes.linewidth': 0.3*scale,
          'font.size': 3*scale,
          'font.weight': 'normal',
          'legend.fontsize': 3*scale,
          'xtick.labelsize': 4*scale,
          'ytick.labelsize': 4*scale,
          'xtick.major.pad': 8,
          'ytick.major.pad': 8,
          'axes.labelpad': 8,
          'text.usetex': True,
          'figure.figsize': [12,8],
          'lines.markersize': 1*scale,
          'lines.linewidth':  0.3*scale,
          'font.family' : 'Times New Roman',
          'mathtext.fontset': 'stix'
          }



"""
Refs.: NewJP 17 015014 (2015)
       JPSJ 84 121001 (2015)
       PRB 99 165430 (2019)
"""

a = 2.456
h = 3.349

try:
    import hankel
    from hankel import HankelTransform 
except:
    print('hankel is not availbel')

def hamiltonian_intralayer(k, lattice, hop_list, E_onsite):
    nsite = len(lattice.sites_cart)
    H = np.zeros([nsite,nsite], dtype=complex)
    for i in range(nsite):
        txi = lattice.sites_cart[i][0:2]
        for end in hop_list[i]:
            p, q, j = end
            txj = lattice.sites_frac[j]
            txj_plus_L = frac2cart([p+txj[0], q+txj[1]],lattice.latt_vec)
            phase = np.exp(1j*np.dot(k,txj_plus_L-txi))
            H[i,j] = H[i,j] + hop_list[i][end]*phase
            H[j,i] = H[j,i] + hop_list[i][end]*np.conj(phase)
    if E_onsite:
        np.fill_diagonal(H, H.diagonal()+E_onsite)
    return H

def hamiltonian_interlayer(k0, k_tld, lattice_tld, k, lattice, Tqs, G_dist_cut):
    nsite = len(lattice.sites_cart)
    nsite_tld = len(lattice_tld.sites_cart)
    H = np.zeros([nsite_tld,nsite], dtype=complex)
    G = k_tld - k0
    G_tld = k - k0
    if np.linalg.norm(k_tld+k)>=G_dist_cut:
        return H
    for i in range(nsite_tld):
        t_X_tld = lattice_tld.sites_cart[i][0:2]
        for j in range(nsite):
            t_X = lattice.sites_cart[j][0:2]
            H[i,j] = Tqs[(i,j)]*np.exp(1j*np.dot(G_tld, t_X_tld)-1j*np.dot(G, t_X))
    return H

def subspace(latt_vec, G_cut):
    bz = BZHexagonal(latt_vec)
    Gs = bz.select_reciprocal_lattice_vectors(G_cut)
    return Gs # including k0 itself
    #return np.array([i for i in Gs if i[0] or i[1]]) ## 12-wave without k0 itself

class Lattice:
    def __init__(self, latt_vec, sites_frac, h):
        self.latt_vec = np.array(latt_vec)
        self.sites_frac = np.array(sites_frac)
        sites_cart = frac2cart(sites_frac, latt_vec)
        self.sites_cart = np.append(sites_cart, [[h]]*len(sites_cart), axis=1)

    def pymatgen_struct(self):
        latt_vec_3d = np.append(self.latt_vec, [[0],[0]], axis=1)
        latt_vec_3d = np.append(latt_vec_3d, [[0,0,20]], axis=0)
        return pmg_struct(latt_vec_3d, ['C']*len(self.sites_cart), self.sites_cart, coords_are_cartesian=True)

    

class IncommenBilayer:

    def __init__(self, theta=30., a=2.456, h=3.349, R=20, dr=0.01, overlap='hole'):
        self.a = a
        self.h = h
        self.sampling_params_hop_func = {'R':R, 'dr':dr}
        latt_vec = np.array([[a*np.cos(np.pi/6), -a*np.sin(np.pi/6)],
                             [a*np.cos(np.pi/6),  a*np.sin(np.pi/6)]])
        if overlap=='hole':
            sites = np.array([[1/3., 1/3.],[2/3., 2/3.]])
        elif overlap=='atom':
            sites = np.array([[0., 0.], [1/3., 1/3.]])
        
        self.lattice_bott = Lattice(latt_vec, sites, 0)
        self.lattice_top = Lattice(rotate_on_vec(theta, latt_vec), sites, h)

    def append_layers(self, layers, slip=1):
        latt_vec_bott = np.append(self.lattice_bott.latt_vec,[[0],[0]], axis=1)
        sites_cart_bott = copy.deepcopy(self.lattice_bott.sites_cart)
        sites_frac_bott = copy.deepcopy(self.lattice_bott.sites_frac)
        latt_vec_top = np.append(self.lattice_top.latt_vec, [[0],[0]], axis=1)
        sites_cart_top = copy.deepcopy(self.lattice_top.sites_cart)
        sites_frac_top = copy.deepcopy(self.lattice_top.sites_frac)
        
        def get_coords2D(layer):
            if layer in ['A','B']:
                latt_vec = latt_vec_bott
                carts = sites_cart_bott if layer=='A' else sites_cart_bott + slip*(latt_vec[0]+latt_vec[1])/3
                fracs = sites_frac_bott if layer=='A' else sites_frac_bott + slip*np.array([1/3., 1/3.])
                return self.lattice_bott, carts, fracs
            elif layer in ['Atld','Btld']:
                latt_vec = latt_vec_top
                carts = sites_cart_top if layer=='Atld' else sites_cart_top + slip*(latt_vec[0]+latt_vec[1])/3
                fracs = sites_frac_top if layer=='Atld' else sites_frac_top + slip*np.array([1/3., 1/3.])
                return self.lattice_top, carts, fracs

        for layer in layers:
            lattice, carts, fracs = get_coords2D(layer)
            for i in layers[layer]:
                z = self.h*i
                carts = copy.deepcopy(carts)
                carts[:,-1] = z
                lattice.sites_cart = np.append(lattice.sites_cart, carts, axis=0)
                lattice.sites_frac = np.append(lattice.sites_frac, fracs, axis=0)

    def choose_hop_func_pz(self, g0=3.12, g1=0.48, a0=2.456/np.sqrt(3), h0=h):
        self.basis_func = 'pz'
        def hop_func(r):
            dr = np.sqrt(r**2 + self.h**2)
            n = self.h/dr
            V_pppi = -g0 * np.exp(2.218 * (a0 - dr))
            V_ppsigma = g1 * np.exp(2.218 * (h0 - dr))
            hop = n**2*V_ppsigma + (1-n**2)*V_pppi
            return hop
        self.hop_func_inter = hop_func
        pmg_st_bott = self.lattice_bott.pymatgen_struct() 
        self.hop_list_intra_bott = calc_hopping_pz_PBC(pmg_st_bott, g0=g0, a0=a0, g1=g1, h0=h0)
        pmg_st_top = self.lattice_top.pymatgen_struct() 
        self.hop_list_intra_top = calc_hopping_pz_PBC(pmg_st_top, g0=g0, a0=a0, g1=g1, h0=h0)
        
    def calc_Tqs(self, k0, ks, ktlds):
        R = self.sampling_params_hop_func['R']
        dr = self.sampling_params_hop_func['dr']
        qs = np.array([ks[i]+ktlds[j]-k0 for i in range(len(ks)) for j in range(len(ktlds))])
        hop_func_FT = hop_func_interlayer_FT(self.hop_func_inter, R=R, dr=dr)
        Tqs = hop_func_FT(qs)
        nsite = len(self.lattice_bott.sites_cart)
        nsite_tld = len(self.lattice_top.sites_cart)
        out = {}
        for i in range(nsite):
            z_i = self.lattice_bott.sites_cart[i][-1]
            for j in range(nsite_tld): 
                z_j = self.lattice_top.sites_cart[j][-1]
                if np.abs(z_i-z_j)<self.h*1.5:
                    out[(i,j)] = Tqs
                else:
                    out[(i,j)] = np.zeros(len(qs))
        return out
 
    def get_k_basis(self, k0, G_cut):
        k0 = np.array(k0)
        Gs_tld = subspace(self.lattice_top.latt_vec, G_cut)
        Gs = subspace(self.lattice_bott.latt_vec, G_cut)
        def get_G_dist():
            G0 = np.array([i for i in Gs_tld if i[0] or i[1]])
            G1 = np.array([i for i in Gs if i[0] or i[1]])
            dG = np.unique(np.round(np.linalg.norm([i-j for i in G0 for j in G1], axis=1), 4))
            G_cut = (dG[0]+dG[1])/2.
            return G_cut
        
        return k0+Gs_tld, k0+Gs, get_G_dist()

    def hamiltonian(self, k0, G_cut=10/2.46, elec_field=0.):
        ks, ks_tld, G_dist_cut = self.get_k_basis(k0, G_cut) ## ks->bottom-layer, ks_tld->top-layer
        Tqs = self.calc_Tqs(k0, ks, ks_tld)
        nk = len(ks)
        nsite = len(self.lattice_bott.sites_cart)
        nk_tld = len(ks_tld)
        nsite_tld = len(self.lattice_top.sites_cart)
        H = np.zeros([nsite*nk+nsite_tld*nk_tld, nsite*nk+nsite_tld*nk_tld], dtype=complex)

        def slice_k(i):
            return slice(nsite*i, nsite*(i+1))

        def slice_ktld(i):
            return slice(nsite*nk+nsite_tld*i,nsite*nk+nsite_tld*(i+1))

        def put_value_in_k(i):
            H[slice_k(i),slice_k(i)] = hamiltonian_intralayer(ks[i], self.lattice_bott, self.hop_list_intra_bott, 0.0)
        tmp = [put_value_in_k(i) for i in range(nk)]
    
        E_onsite = elec_field*self.h
        def put_value_in_k_tld(i):
            H[slice_ktld(i), slice_ktld(i)] = hamiltonian_intralayer(ks_tld[i], self.lattice_top, self.hop_list_intra_top, E_onsite)
        tmp = [put_value_in_k_tld(i) for i in range(nk_tld)]
    
        def put_value_between_k_and_k_tld(i,j):
            Tqsij = {pair:Tqs[pair][nk_tld*i+j] for pair in Tqs}
            Hij = hamiltonian_interlayer(k0, ks[i], self.lattice_bott, ks_tld[j], self.lattice_top, Tqsij, G_dist_cut)
            H[slice_k(i), slice_ktld(j)] = Hij
            H[slice_ktld(j), slice_k(i)] = np.matrix(Hij).H
        tmp = [put_value_between_k_and_k_tld(i,j) for i in range(nk) for j in range(nk_tld)]
        return H        

def hop_func_interlayer_FT(hop_func, R=5., dr=0.01):
    """
    the Fourier transform of interlayer hopping
    it can be transformed to be Hankel transform  
    tq = 1/s * int_0^infinity T(r)J0(kr)r dr
    """
    def hop_func_FT(q):
        try:
            q = np.linalg.norm(q, axis=1)
        except:
            q = np.linalg.norm(q)
        ht = HankelTransform(nu=0, N=round(R/dr), h=dr)
        return 2*np.pi * ht.transform(hop_func, q, ret_err=False)/(a**2*np.sin(np.pi/3))
    return hop_func_FT

def calc_quasi_band(G_cut=10/2.46, dk=0.01, elec_fields=[0.], append={}, slip=1, overlap='hole', theta=30.):
    tbg30 = IncommenBilayer(overlap=overlap,theta=theta)
    tbg30.append_layers(append, slip=slip)
    tbg30.choose_hop_func_pz()
    bz = BZHexagonal(tbg30.lattice_bott.latt_vec)
    K = bz.special_points()['K'][0]
    print(K)
    ks, inds = kpoints_line_mode([[-0.415,0],[0.415,0]], dk=dk)
    #ks, inds = kpoints_line_mode([[-0.401,0.0],[0.401, 0.0]], dk=dk)
    for ef in elec_fields:
        vals = []
        for k0 in ks:
            H = tbg30.hamiltonian(k0, G_cut, ef)
            val, vec, info = zheev(H, 0)
            if info:
                raise ValueError('zheev failed!')
            vals.append(val)
        np.savez_compressed('EIGvals_elec_field_%s' % ef, vals=vals)

def calc_vec_onek(G_cut=10/2.46, k0=[0,0], elec_fields=[0.], append={}, overlap='hole', slip=1):
    tbg30 = IncommenBilayer(overlap=overlap)
    tbg30.append_layers(append, slip=slip)
    tbg30.choose_hop_func_pz()
    for ef in elec_fields:
        H = tbg30.hamiltonian(k0, G_cut, ef)
        ks, ks_tld, G_dist_cut = tbg30.get_k_basis(k0, G_cut)
        val, vec= eigh(H)
        np.savez_compressed('EIGvec_elec_field_%s' % ef, val=val, vec=vec, k_basis=[ks, ks_tld])

def plot_quasi_band(eig_f='EIGvals.npz', xlim=None, ylim=[-2.7,0.1],show=True):
    data = np.load(eig_f)
    vals = data['vals']
    nk = len(vals)
    nb = len(vals[0])
    ks = np.array(range(nk))*0.01
    ind = int(nb/2)
    ef = (np.max(vals[:,ind-1]) + np.min(vals[:,ind]))/2
    vals = vals - ef
    from matplotlib import pyplot as plt
    from tBG import params
    plt.rcParams.update(params)
    for i in range(nb):
        plt.plot(ks, vals[:,i], color='black')
    if xlim is not None:
        plt.xlim(xlim)
    plt.ylim(ylim)
    if show:
        plt.show()
    else:
        plt.savefig('quasi_band.pdf')
    plt.close()

def get_coords(lattice, L):
    nsite = len(lattice.sites_frac)
    mat = lattice.latt_vec.T
    ms = []
    ns = []
    def get_ind_limit(sublatt):
        frac = lattice.sites_frac[sublatt]
        c = np.matmul(frac, lattice.latt_vec)
        for i in [-L, L]:
            for j in [-L,L]:
                c_new = np.array([i,j])-c
                m,n = np.linalg.solve(mat, c_new)
                ms.append(int(m))
                ns.append(int(n))
        return min(ms), max(ms), min(ns), max(ns)
    coords = []
    for s in range(nsite):
        frac = lattice.sites_frac[s]
        z = lattice.sites_cart[s][-1]
        m0, m1, n0, n1 = get_ind_limit(s)
        fracs = np.array([[i+frac[0], j+frac[1]] for i in range(m0, m1+1) for j in range(n0,n1+1)])
        coords_i = frac2cart(fracs, lattice.latt_vec)
        coords_i = np.append(coords_i, [[z]]*len(coords_i), axis=1)
        inds0 = np.where(coords_i[:,0]>=-L)[0]
        inds1 = np.where(coords_i[:,0]<=L)[0]
        inds2 = np.where(coords_i[:,1]>=-L)[0]
        inds3 = np.where(coords_i[:,1]<=L)[0]
        inds = reduce(np.intersect1d, (inds0, inds1, inds2, inds3))
        coords.append(coords_i[inds])
    return coords

def get_mc_line(coords, color, bond_length=1.6):
    coords = copy.deepcopy(coords)
    nsite = len(coords)
    xmin, ymin, zmin = np.min(coords, axis=0)
    xmax, ymax, zmax = np.max(coords, axis=0)
    #coords[:,0] = coords[:,0] - xmin + 5
    #coords[:,1] = coords[:,1] - ymin + 5
    latt_vec = np.array([[xmax-xmin+100, 0, 0], [0, ymax-ymin+100, 0], [0,0,100]])
    pmg_st = pmg_struct(latt_vec, ['C']*nsite, coords, coords_are_cartesian=True)
    neigh_list = pmg_st.get_neighbor_list(bond_length)
    neigh_list = filter_neig_list(neigh_list)
    begs = neigh_list[0]
    ends = neigh_list[1]
    line = mc.LineCollection([[coords[begs[i]][0:2], coords[ends[i]][0:2]] for i in range(len(begs))], 0.1, colors=color)
    return line

def _plot_states_before_hybridization():
    ks = np.array([[[-2.55414037, -1.47463363],
        [-2.55414037,  1.47463363],
        [ 0.        , -2.94926726],
        [ 0.        ,  2.94926726],
        [ 2.55414037, -1.47463363],
        [ 2.55414037,  1.47463363]],

       [[-2.94926726,  0.        ],
        [-1.47463363,  2.55414037],
        [-1.47463363, -2.55414037],
        [ 1.47463363,  2.55414037],
        [ 1.47463363, -2.55414037],
        [ 2.94926726,  0.        ]]])

    alpha = 1/np.sqrt(12)*np.array([-1, 1, 1, -1, 1,-1, -1,1, -1,1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
    beta =  1/np.sqrt(12)*np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 1, -1, 1,-1, -1,1, -1,1, 1, -1], dtype=complex)
    lambda_ = 1/np.sqrt(12)*np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0], dtype=complex)
    mu = 1/np.sqrt(12)*np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1], dtype=complex)
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(2,4, figsize=[20,10])
    plot_vec(axes[0,0], alpha, ks, L=25) 
    axes[0,0].set_title('$\\alpha$')
    plot_vec(axes[0,1], beta, ks, L=25) 
    axes[0,1].set_title('$\\beta$')

    plot_vec(axes[0,2], 1/np.sqrt(2)*(alpha+beta), ks, L=25) 
    axes[0,2].set_title('$\\alpha+\\beta$')
    plot_vec(axes[0,3], 1/np.sqrt(2)*(alpha-beta), ks, L=25) 
    axes[0,3].set_title('$\\alpha-\\beta$')


    plot_vec(axes[1,0], lambda_, ks, L=25) 
    axes[1,0].set_title('$\lambda$')
    plot_vec(axes[1,1], mu, ks, L=25) 
    axes[1,1].set_title('$\mu$')

    plot_vec(axes[1,2], 1/np.sqrt(2)*(lambda_+mu), ks, L=25) 
    axes[1,2].set_title('$\lambda+\mu$')
    plot_vec(axes[1,3], 1/np.sqrt(2)*(lambda_-mu), ks, L=25) 
    axes[1,3].set_title('$\lambda-\mu$')

    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.savefig('states_before_hybrid.png', bbox_inches='tight', pad_inches=0.)


def plot_vec(ax, vec, ks, tbg, L=25):
    #tbg = IncommenBilayer(overlap=overlap)
    #tbg.append_layers(append,slip=slip)
    nsite_bott = len(tbg.lattice_bott.sites_cart)
    nsite_top = len(tbg.lattice_top.sites_cart)
    coords_bott = get_coords(tbg.lattice_bott, L)
    line_bott = get_mc_line(np.concatenate(coords_bott), 'blue')
    ax.add_collection(line_bott)
    coords_top = get_coords(tbg.lattice_top, L)
    line_top = get_mc_line(np.concatenate(coords_top), 'red')
    ax.add_collection(line_top)
    nk = len(ks[0])
    coords = [coords_bott, coords_top]
    cs = [['blue','green'],['red','yellow']]
    for i in [0,1]: # 0 for bottom 1 for top
        coord = coords[i] ## coord of lattice 0 
        for X in range(len(coord)):
            coordX = coord[X][:,0:2] # coords of X sublattice
            CX = 0.
            for ik in range(nk):
                k = ks[i][ik]
                vec_tmp = vec[ik*nsite_bott+X] if i==0 else vec[nk*nsite_bott+ik*nsite_top+X]
                CX = CX + vec_tmp*np.exp(1j*np.dot(coordX,k))
            CX = np.square(np.abs(CX))
            ax.scatter(coordX[:,0], coordX[:,1], s=CX*800, color=cs[i][int(X/2)], alpha=0.6, linewidths=0)
    ax.set_xlim(-L,L)
    ax.set_ylim(-L,L)
    ax.axis('equal')
    ax.axis('off')

def plot_vec_from_file(nb, eig_f='EIFvec.npz', L=25, title='', tbg=None):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1,1, figsize=[10, 10])

    data = np.load(eig_f)
    val = data['val']
    vec = data['vec'][:,nb]

    ks = data['k_basis']

    plot_vec(ax, vec, ks, tbg, L=L)
    fig.suptitle(title)
    plt.tight_layout(pad=0,h_pad=0, w_pad=0)
    plt.savefig('vec_nb%s.png' % nb, bbox_inches='tight', pad_inches=0)
    plt.close()
    
def plot_vec_from_file_combtwo(eig_f='EIFvec.npz', L=25, nb_th=[1,2], title='', tbg=None):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=[20, 10])

    data = np.load(eig_f)
    val = data['val']
    vec0 = data['vec'][:,nb_th[0]]
    vec1 = data['vec'][:,nb_th[1]]
    veci = (vec0+vec1)/np.sqrt(2)
    vecj = (vec0-vec1)/np.sqrt(2)

    ks = data['k_basis']

    plot_vec(axes[0], veci, ks, tbg, L=L)
    plot_vec(axes[1], vecj, ks, tbg, L=L)
    fig.suptitle(title)
    plt.tight_layout(pad=0,h_pad=0, w_pad=0)
    plt.savefig('vec_nb_%s-%s.png' % (nb_th[0], nb_th[1]), bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_chg_from_file_plustwo(eig_f='EIFvec.npz', L=25, nb_th=[1,2], title='', tbg=None):
    from matplotlib import pyplot as plt
    plt.rcParams.update(params)
    nsite_bott = len(tbg.lattice_bott.sites_cart)
    nsite_top = len(tbg.lattice_top.sites_cart)
    fig, ax = plt.subplots(1,1, figsize=[10, 10])
    coords_bott = get_coords(tbg.lattice_bott, L)
    line_bott = get_mc_line(np.concatenate(coords_bott), 'blue')
    ax.add_collection(line_bott)
    coords_top = get_coords(tbg.lattice_top, L)
    line_top = get_mc_line(np.concatenate(coords_top), 'red')
    ax.add_collection(line_top)

    data = np.load(eig_f)
    val = data['val']
    vec0 = data['vec'][:,nb_th[0]]
    vec1 = data['vec'][:,nb_th[1]]

    ks = data['k_basis']
    nb = len(val)
    nk = len(ks[0])

    coords = [coords_bott, coords_top]
    cs = [['blue', 'green'],['red','yellow']]
    for i in [0,1]:
        coord = coords[i]
        for X in range(len(coords[i])):
            coordX = coord[X][:,0:2]
            CX0 = 0.
            CX1 = 0.
            for ik in range(nk):
                k = ks[i][ik]
                vec0_tmp = vec0[ik*nsite_bott+X] if i==0 else vec0[nk*nsite_bott+ik*nsite_top+X]
                vec1_tmp = vec1[ik*nsite_bott+X] if i==0 else vec1[nk*nsite_bott+ik*nsite_top+X]
                CX0 = CX0 + vec0_tmp*np.exp(1j*np.dot(coordX,k))
                CX1 = CX1 + vec1_tmp*np.exp(1j*np.dot(coordX,k))
            CX0 = np.square(np.abs(CX0))
            CX1 = np.square(np.abs(CX1))
            CX = (CX0+CX1)/2
            ax.scatter(coordX[:,0], coordX[:,1], s=CX*800, color=cs[i][int(X/2)], alpha=0.6, linewidths=0)
    ax.axis('off')
    ax.axis('equal',adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xlim(-L,L)
    ax.set_ylim(-L,L)
    plt.tight_layout(pad=0,h_pad=0, w_pad=0)
    plt.savefig('chg_nb_%s-%s.png' % (nb_th[0], nb_th[1]), bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    t0 = time.time()
    main(G_cut=10/2.46, hop_type='wannier', P=10, dk=0.01)
    t1 = time.time()
    print('total time: %.0f s' % (t1-t0))
