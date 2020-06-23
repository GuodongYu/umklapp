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

"""
Refs.: NewJP 17 015014 (2015)
       JPSJ 84 121001 (2015)
       PRB 99 165430 (2019)
"""

a = 2.46
h = 3.35

try:
    import hankel
    from hankel import HankelTransform 
except:
    print('hankel is not availbel')

def hamiltonian_intralayer(k, lattice, hop_list, E_onsite):
    H = np.zeros([2,2], dtype=complex)
    for i in [0,1]:
        txi = lattice.sites_cart[i]
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
    H = np.zeros([2,2], dtype=complex)
    G = k_tld - k0
    G_tld = k - k0
    if np.linalg.norm(k_tld+k)>=G_dist_cut:
        return H
    for i in [0,1]:
        t_X_tld = lattice_tld.sites_cart[i]
        for j in [0,1]:
            t_X = lattice.sites_cart[j]
            H[i,j] = Tqs[(i,j)]*np.exp(1j*np.dot(G_tld, t_X_tld)-1j*np.dot(G, t_X))
    return H

def subspace(latt_vec, G_cut):
    bz = BZHexagonal(latt_vec)
    Gs = bz.select_reciprocal_lattice_vectors(G_cut)
    #return Gs
    return np.array([i for i in Gs if i[0] or i[1]])

def determine_k_l(qs, R):
    k = np.round(qs[:,0]/(np.pi/R))
    l = np.round(qs[:,1]/(np.pi/R))
    return np.array(k, dtype=int),np.array(l, dtype=int)

class Lattice:
    def __init__(self, latt_vec, sites_frac):
        self.latt_vec = np.array(latt_vec)
        self.sites_frac = np.array(sites_frac)
        self.sites_cart = frac2cart(sites_frac, latt_vec)
        self.vec_to_NN = np.array([(latt_vec[0]+latt_vec[1])/3, -(latt_vec[0]+latt_vec[1])/3])

    def pymatgen_struct(self):
        latt_vec_3d = np.append(self.latt_vec, [[0],[0]], axis=1)
        latt_vec_3d = np.append(latt_vec_3d, [[0,0,100]], axis=0)
        coords_3d = np.append(self.sites_cart, [[0],[0]], axis=1)
        return pmg_struct(latt_vec_3d, ['C','C'], coords_3d, coords_are_cartesian=True)

class IncommenBilayer:

    def __init__(self, theta=30., a=2.46, h=3.35, R=5, dr=0.005):
        self.a = 2.46
        self.h = 3.35
        self.sampling_params_hop_func = {'R':R, 'dr':dr}
        latt_vec = np.array([[a*np.cos(np.pi/6), -a*np.sin(np.pi/6)],
                             [a*np.cos(np.pi/6),  a*np.sin(np.pi/6)]])
        sites = np.array([[1/3., 1/3.],[2/3., 2/3.]])
        self.lattice_bott = Lattice(latt_vec, sites)
        self.lattice_top = Lattice(rotate_on_vec(theta, latt_vec), sites)

    def choose_hop_func_pz(self, g0=2.7, g1=0.48, a0=2.46/np.sqrt(3), h0=3.35):
        self.basis_func = 'pz'
        def hop_func(r):
            dr = np.sqrt(r**2 + self.h**2)
            n = self.h/dr
            V_pppi = -g0 * np.exp(2.218 * (a0 - dr))
            V_ppsigma = g1 * np.exp(2.218 * (h0 - dr))
            hop = n**2*V_ppsigma + (1-n**2)*V_pppi
            return hop
        self.hop_func_inter = hop_func
        pmg_st = self.lattice_bott.pymatgen_struct() ## lattice_bott and lattice_top give the same hop_list
        self.hop_list_intra = calc_hopping_pz_PBC(pmg_st, g0=g0, a0=a0, g1=g1, h0=h0)

    def choose_hop_func_wannier(self, P=0):
        delta = 0.1048*np.log(1+P/5.73)
        self.h = (1-delta)*self.h
        self.basis_func = 'wannier'
        lambda0, xi0, k0, lambda3, xi3, x3, lambda6, xi6, x6, k6= \
                                       hop_params_wannier_interlayer(P)
        hop_func = {}
        for i in [0,1]:
            vec_to_NN_i = self.lattice_bott.vec_to_NN[i]
            for j in [0,1]:
                vec_to_NN_j = self.lattice_top.vec_to_NN[j]
                hop_func[(i,j)] = hop_func_wannier_interlayer(vec_to_NN_i, vec_to_NN_j, lambda0, xi0, k0, \
                                                                lambda3, xi3, x3, lambda6, xi6, x6, k6, a)
        self.hop_func_inter = hop_func
        self.hop_list_intra = hop_list_graphene_wannier()
        print('go into Tqs maxtrix')
        self.Tqs_matrix = self.calc_FT_wannier()
        print('Tqs matrix done')

    def calc_FT_wannier(self):
        R = self.sampling_params_hop_func['R']
        dr = self.sampling_params_hop_func['dr']
        n = int(2*R/dr)+1
        print(n)
        rs = np.array([[-R+dr*i, -R+dr*j] for i in range(n) for j in range(n)])
        refs = np.array([[0.0, 0.0]]*len(rs))
        Tqs = {}
        ds = dr*dr
        s = (a**2*np.sin(np.pi/3))
        C_mat = np.zeros([n,n], dtype=complex)
        ind_plus = int(n/2) ## number of positive frequency
        ind_minus = n-int(n/2)-1 ## number of negative frequency (-1 for zero frequency)
        def new(i,j):
            i = i if i <=ind_plus else -n+i
            j = j if j <=ind_plus else -n+j
            return i, j 
        def put_val_plus_plus(i,j):
            C_mat[i,j] = np.exp(1j*np.pi*(i+j))
        tmp = [put_val_plus_plus(i,j) for i in range(ind_plus+1) for j in range(ind_plus+1)]
        del tmp
        def put_val_plus_minus(i,j):
            C_mat[i,j] = np.exp(1j*np.pi*(i+j-n))
        tmp = [put_val_plus_minus(i,j) for i in range(ind_plus+1) for j in range(ind_plus+1,n)]
        del tmp
        def put_val_minus_plus(i,j):
            C_mat[i,j] = np.exp(1j*np.pi*(i+j-n))
        tmp = [put_val_minus_plus(i,j) for i in range(ind_plus+1,n) for j in range(ind_plus+1)]
        del tmp
        def put_val_minus_minus(i,j):
            C_mat[i,j] = np.exp(1j*np.pi*(i+j-2*n))
        tmp = [put_val_minus_minus(i,j) for i in range(ind_plus+1,n) for j in range(ind_plus+1,n)]
        del tmp
        for i in [(0,0),(0,1),(1,0),(1,1)]:
            hop_func = self.hop_func_inter[i]
            trs = hop_func(rs, refs)
            trs = trs.reshape(n,n)
            print('fftw run')
            Tqs[i] = ds/s * C_mat * np.fft.fft2(trs)
            print('fftw done')
        return Tqs
        
    def calc_Tqs(self, k0, ks, ktlds):
        R = self.sampling_params_hop_func['R']
        dr = self.sampling_params_hop_func['dr']
        qs = np.array([ks[i]+ktlds[j]-k0 for i in range(len(ks)) for j in range(len(ktlds))])
        if self.basis_func == 'pz':
            hop_func_FT = hop_func_interlayer_FT(self.hop_func_inter, R=R, dr=dr)
            Tqs = hop_func_FT(qs)
            return {(0,0):Tqs, (0,1):Tqs, (1,0):Tqs, (1,1):Tqs}
        elif self.basis_func == 'wannier':
            ks, ls = determine_k_l(qs, R)
            Tqs = {}
            for XXt in [(0,0),(0,1),(1,0),(1,1)]:
                Tqs[XXt]=np.array([self.Tqs_matrix[XXt][ks[i],ls[i]] for i in range(len(ks))])
            return Tqs
    
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
        ks, ks_tld, G_dist_cut = self.get_k_basis(k0, G_cut)
        Tqs = self.calc_Tqs(k0, ks, ks_tld)
        nk = len(ks)
        nk_tld = len(ks_tld)
        H = np.zeros([2*(nk+nk_tld), 2*(nk+nk_tld)], dtype=complex)
        def put_value_in_k(i):
            H[2*i:2*(i+1),2*i:2*(i+1)] = hamiltonian_intralayer(ks[i], self.lattice_bott, self.hop_list_intra, 0.0)
        tmp = [put_value_in_k(i) for i in range(nk)]
        del tmp
    
        E_onsite = elec_field*self.h
        def put_value_in_k_tld(i):
            H[2*i:2*(i+1),2*i:2*(i+1)] = hamiltonian_intralayer(ks_tld[i-nk], self.lattice_top, self.hop_list_intra, E_onsite)
        tmp = [put_value_in_k_tld(i) for i in range(nk, nk+nk_tld)]
        del tmp
    
        def put_value_between_k_and_k_tld(i,j):
            Tqsij = {k:Tqs[k][nk*i+j-nk] for k in Tqs}
            Hij = hamiltonian_interlayer(k0, ks[i], self.lattice_bott, ks_tld[j-nk], self.lattice_top, Tqsij, G_dist_cut)
            H[2*i:2*(i+1), 2*j:2*(j+1)] = Hij
            H[2*j:2*(j+1), 2*i:2*(i+1)] = np.matrix(Hij).H
        tmp = [put_value_between_k_and_k_tld(i,j) for i in range(nk) for j in range(nk,nk+nk_tld)]
        del tmp
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

def calc_quasi_band(G_cut=10/2.46, dk=0.01, hop_type='wannier', P=0, elec_fields=[0.]):
    tbg30 = IncommenBilayer()
    if hop_type == 'wannier':
        tbg30.choose_hop_func_wannier(P=P)
    elif hop_type == 'pz':
        tbg30.choose_hop_func_pz()
    bz = BZHexagonal(tbg30.lattice_bott.latt_vec)
    K = bz.special_points()['K'][0]
    ks, inds = kpoints_line_mode([K,-K], dk=dk)
    for ef in elec_fields:
        vals = []
        for k0 in ks:
            H = tbg30.hamiltonian(k0, G_cut, ef)
            val, vec, info = zheev(H, 0)
            if info:
                raise ValueError('zheev failed!')
            vals.append(val)
        np.savez_compressed('EIGvals_elec_field_%s' % ef, vals=vals)

def calc_vec_onek(G_cut=10/2.46, hop_type='pz', P=0, k0=[0,0], elec_fields=[0.]):
    if hop_type == 'wannier':
        tbg30 = IncommenBilayer(R=200, dr=0.04)
        tbg30.choose_hop_func_wannier(P=P)
        #k0 = [0,0]
    elif hop_type == 'pz':
        tbg30 = IncommenBilayer(R=5, dr=0.005)
        tbg30.choose_hop_func_pz()
        #k0 = [10**(-10),10**(-10)]
    for ef in elec_fields:
        H = tbg30.hamiltonian(k0, G_cut, ef)
        ks, ks_tld, G_dist_cut = tbg30.get_k_basis(k0, G_cut)
        val, vec= eigh(H)
        np.savez_compressed('EIGvec_elec_field_%s' % ef, val=val, vec=vec, k_basis=[ks, ks_tld])

def plot_quasi_band(eig_f='EIGvals.npz', xlim=None, ylim=[-2.7,0.1]):
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
    plt.show()
    plt.close()

def get_coords(lattice, L):
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
    for s in [0,1]:
        frac = lattice.sites_frac[s]
        m0, m1, n0, n1 = get_ind_limit(s)
        fracs = np.array([[i+frac[0], j+frac[1]] for i in range(m0, m1+1) for j in range(n0,n1+1)])
        coords_i = frac2cart(fracs, lattice.latt_vec)
        inds0 = np.where(coords_i[:,0]>=-L)[0]
        inds1 = np.where(coords_i[:,0]<=L)[0]
        inds2 = np.where(coords_i[:,1]>=-L)[0]
        inds3 = np.where(coords_i[:,1]<=L)[0]
        inds = reduce(np.intersect1d, (inds0, inds1, inds2, inds3))
        coords.append(coords_i[inds])
    return coords

def get_mc_line(coords, color, bond_length=2.0):
    coords = copy.deepcopy(coords)
    nsite = len(coords)
    xmin, ymin = np.min(coords, axis=0)
    xmax, ymax = np.max(coords, axis=0)
    #coords[:,0] = coords[:,0] - xmin + 5
    #coords[:,1] = coords[:,1] - ymin + 5
    coords = np.append(coords, [[0]]*nsite, axis=1)
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


def plot_vec(ax, vec, ks, L=25):
    tbg = IncommenBilayer()
    coords_bott = get_coords(tbg.lattice_bott, L)
    line_bott = get_mc_line(np.concatenate(coords_bott), 'blue')
    ax.add_collection(line_bott)
    coords_top = get_coords(tbg.lattice_top, L)
    line_top = get_mc_line(np.concatenate(coords_top), 'red')
    ax.add_collection(line_top)
    nk = len(ks[0])
    coords = [coords_bott, coords_top]
    cs = ['blue','red']
    for i in [0,1]:
        coord = coords[i]
        for X in [0,1]:
            coordX = coord[X]
            CX = 0.
            for ik in range(nk):
                k = ks[i][ik]
                CX = CX + vec[i*2*nk+ik*2+X]*np.exp(1j*np.dot(coordX,k))
            CX = np.square(np.abs(CX))
            ax.scatter(coordX[:,0], coordX[:,1], s=CX*70, color=cs[i], alpha=0.6, linewidths=0)
    ax.set_xlim(-L,L)
    ax.set_ylim(-L,L)
    ax.axis('equal')
    ax.axis('off')

def plot_vec_from_file_new(eig_f='EIFvec.npz', L=25, nb_th=[1,2], title=''):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=[10, 5])

    data = np.load(eig_f)
    val = data['val']
    vec0 = data['vec'][:,nb_th[0]]
    vec1 = data['vec'][:,nb_th[1]]

    ks = data['k_basis']

    plot_vec(axes[0], vec0, ks, L=L)
    plot_vec(axes[1], vec1, ks, L=L)
    fig.suptitle(title)
    plt.tight_layout(pad=0,h_pad=0, w_pad=0)
    plt.savefig('vec.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
def plot_vec_from_file(eig_f='EIFvec.npz', L=25, nb_th=[1,2], title=''):
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=[10, 5])
    tbg = IncommenBilayer()
    coords_bott = get_coords(tbg.lattice_bott, L)
    line_bott = get_mc_line(np.concatenate(coords_bott), 'blue')
    axes[0].add_collection(line_bott)
    line_bott = get_mc_line(np.concatenate(coords_bott), 'blue')
    axes[1].add_collection(line_bott)
    coords_top = get_coords(tbg.lattice_top, L)
    line_top = get_mc_line(np.concatenate(coords_top), 'red')
    axes[0].add_collection(line_top)
    line_top = get_mc_line(np.concatenate(coords_top), 'red')
    axes[1].add_collection(line_top)

    data = np.load(eig_f)
    val = data['val']
    vec0 = data['vec'][:,nb_th[0]]
    vec1 = data['vec'][:,nb_th[1]]

    ks = data['k_basis']
    nb = len(val)
    nk = int(nb/4)

    coords = [coords_bott, coords_top]
    cs = ['blue','red']
    for i in [0,1]:
        coord = coords[i]
        for X in [0,1]:
            coordX = coord[X]
            CX0 = 0.
            CX1 = 0.
            for ik in range(nk):
                k = ks[i][ik]
                CX0 = CX0 + vec0[i*2*nk+ik*2+X]*np.exp(1j*np.dot(coordX,k))
                CX1 = CX1 + vec1[i*2*nk+ik*2+X]*np.exp(1j*np.dot(coordX,k))
            CX0 = np.square(np.abs(CX0))
            CX1 = np.square(np.abs(CX1))
            CX = CX0+CX1
            axes[0].scatter(coordX[:,0], coordX[:,1], s=CX0*70, color=cs[i], alpha=0.6, linewidths=0)
            axes[1].scatter(coordX[:,0], coordX[:,1], s=CX1*70, color=cs[i], alpha=0.6, linewidths=0)
    axes[0].set_xlim(-L,L)
    axes[1].set_xlim(-L,L)
    axes[0].set_ylim(-L,L)
    axes[1].set_ylim(-L,L)
    axes[0].axis('equal')
    axes[0].axis('off')
    axes[1].axis('equal')
    axes[1].axis('off')
    fig.suptitle(title)
    plt.tight_layout(pad=0,h_pad=0, w_pad=0)
    plt.savefig('vec.png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_vec_twoplus(eig_f='EIFvec.npz', L=25, nb_th=[1,2], title=''):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(1,1)
    tbg = IncommenBilayer()
    coords_bott = get_coords(tbg.lattice_bott, L)
    coords_top = get_coords(tbg.lattice_top, L)
    line_bott = get_mc_line(np.concatenate(coords_bott), 'blue')
    ax.add_collection(line_bott)
    line_bott = get_mc_line(np.concatenate(coords_bott), 'blue')
    line_top = get_mc_line(np.concatenate(coords_top), 'red')
    ax.add_collection(line_top)

    data = np.load(eig_f)
    val = data['val']
    vec0 = data['vec'][:,nb_th[0]]
    vec1 = data['vec'][:,nb_th[1]]
    vec = vec0+vec1
    ks = data['k_basis']
    nb = len(val)
    nk = int(nb/4)

    coords = [coords_bott, coords_top]
    cs = ['blue','red']
    for i in [0,1]:
        coord = coords[i]
        for X in [0,1]:
            coordX = coord[X]
            CX = 0.
            for ik in range(nk):
                k = ks[i][ik]
                CX = CX + vec[i*2*nk+ik*2+X]*np.exp(1j*np.dot(coordX,k))
            CX = np.square(np.abs(CX))
            ax.scatter(coordX[:,0], coordX[:,1], s=CX*70, color=cs[i], alpha=0.6, linewidths=0)
    ax.set_xlim(-L,L)
    ax.set_ylim(-L,L)
    ax.axis('equal')
    ax.axis('off')
    fig.suptitle(title)
    plt.tight_layout(pad=0,h_pad=0, w_pad=0)
    plt.show()
    #plt.savefig('vec.png', bbox_inches='tight', pad_inches=0)
    plt.close()
if __name__ == '__main__':
    t0 = time.time()
    main(G_cut=10/2.46, hop_type='wannier', P=10, dk=0.01)
    t1 = time.time()
    print('total time: %.0f s' % (t1-t0))
