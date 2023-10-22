import proplot as plt
import numpy as np
from scipy import linalg
from scipy.sparse import kron, csr_matrix, identity

from pyqed.oqs import Redfield_solver

from pyqed import pauli, tensor, multi_spin, raising, lowering,\
    coth, transform, dag, boson, destroy, sort

from pyqed.superoperator import dm2vec, left, right
from pyqed.units import au2ev, au2wavenumber, au2k, au2wavenumber, au2fs


def bose_dist(w, beta):
    return 1./(np.exp(beta * w) - 1.)


def ohmic_spectral_density(w):
    wc = 700/au2wavenumber
    eta = 0.9
    return eta * w * np.exp(-w/wc)


def ohmic(w, T=300.):
    beta = au2k/T
    u = np.exp(beta * w)
    eta = 0.9

    if w == 0.:
        return eta/beta
    elif w > 0:
        return ohmic_spectral_density(w) * (bose_dist(w, beta) + 1)
    elif w < 0:
        return ohmic_spectral_density(-w) * bose_dist(-w, beta)


def drude_spectral_density(w, lamd=800/au2wavenumber, gams=150/au2wavenumber):
    return lamd * gams * w / (gams**2 + w**2)


def drude(w, T=300.):
    beta = au2k/T
    lamd = 800/au2wavenumber
    gams = 150/au2wavenumber

    if w == 0.:
        return lamd/(beta*gams)
    elif w > 0:
        return drude_spectral_density(w, lamd=lamd, gams=gams) * (bose_dist(w, beta) + 1)
    elif w < 0:
        return drude_spectral_density(-w, lamd=lamd, gams=gams) * bose_dist(-w, beta)


def jump(f, i, d=2):
    J = np.zeros((d, d))
    J[f, i] = 1.
    return csr_matrix(J)


def frenkel(nsites, E, J, d=2):
    if d != 2:
        raise NotImplementedError('local dimension d can only be 2.')

    idm = np.identity(d)

    if nsites == 2:
        H = E * (kron(ee, idm) + kron(idm, ee)) + \
            J * (kron(sp, sm) + kron(sm, sp))
    return H


def holstein(E, J, freq_vib, hr, nsite=2, nel=2, nvib=2):
    idm = identity(nel)
    idv = identity(nvib)  # vibrational identity
    idm_ev = identity(nel * nvib)

    hvib = boson(freq_vib, nvib, ZPE=True)

    a = destroy(nvib)

    H1 = E * kron(ee, idv) + kron(idm, hvib) + hr * kron(ee, a + dag(a))
    H2 = H1.copy()

    sm = lowering()
    Sm = kron(sm, idv)
    Sp = dag(Sm)

    H = kron(H1, idm_ev) + kron(idm_ev, H2) + J * (kron(Sp, Sm) + kron(Sm, Sp))

    return H


# fig, ax = plt.subplots()
# w = np.linspace(-40000/au2wavenumber, 40000/au2wavenumber)
# ax.plot(w, [drude(x) for x in w])
# plt.show()

s0, sx, sy, sz = pauli()
sp = raising()
sm = lowering()
gg = jump(0, 0)
ee = jump(1, 1)

J = 870/au2wavenumber
hr = 834/au2wavenumber
freq_vib = 1230/au2wavenumber
nel = 2
nvib = 2
idm_ev = identity(nel * nvib)
idv = identity(nvib)

H = holstein(E=18950/au2wavenumber, J=J, freq_vib=freq_vib, hr=hr, nvib=nvib)
u = H.shape[-1]
evals, ref = linalg.eigh(H.toarray())
print('size of Hilbert space', H.shape)

c_ops = [kron(kron(ee, idv), idm_ev) + kron(idm_ev, kron(ee, idv))]
spectra = [drude]
edip = kron(kron(sx, idv), idm_ev) + kron(idm_ev, kron(sx, idv))

beta = au2k/300.
hvib = (boson(freq_vib, nvib, ZPE=True)).toarray()
equ_mat = linalg.expm(-beta * hvib) / np.trace(linalg.expm(-beta * hvib))
rho0 = kron(kron(gg, equ_mat), kron(gg, equ_mat))
print(rho0.shape)
print(rho0.trace())

sol = Redfield_solver(H, c_ops=c_ops, spectra=spectra)
R, evecs = sol.redfield_tensor()
print('Redfield tensor done ..., shape = {}'.format(R.shape))

dt = 0.1/au2fs
Nt = 100
print('time steps = {} and time interval = {}'.format(Nt, dt * au2fs))

ts = np.arange(Nt) * dt
U = sol.propagator(t=ts, method='eseries')
print("propagator done ...")

expect = sol.expect(rho0=(edip @ rho0).copy(), e_ops=[edip])

fig, ax = plt.subplots()
ax.plot(ts, expect[:, 0].real)
fig.savefig('obs.png')

np.savez('propagator', U)
np.savez('obs', expect)

# edip = transform(edip, evecs)
# u = np.triu(edip)
# l = np.tril(edip)
# print(u[1, 0], l[0, 1])
# print('Both numbers should vanish.')
# print(u[1, 0], l[0, 1])
# print('Both numbers should vanish.')
# rho0_eb = transform(rho0, evecs)

# corr1 = sol.correlation_4op_3t(
#     rho0=rho0_eb.copy(), oplist=[l, u, u, l], signature='lllr', tau=ts)
# corr2 = sol.correlation_4op_3t(
#     rho0=rho0_eb.copy(), oplist=[l, u, u, l], signature='lrlr', tau=ts)
# corr3 = sol.correlation_4op_3t(
#     rho0=rho0_eb.copy(), oplist=[l, u, u, l], signature='llrr', tau=ts)

# np.savez('corr_J{:.2f}'.format(J * au2wavenumber), corr1, corr2, corr3)
