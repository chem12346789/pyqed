'''
This example shows how to use the DEOM class to simulate the dynamics of a two-level system coupled to a drude bath.
'''
from functools import reduce
import itertools
from pyqed.mol import Mol, LVC, Mode
from pyqed.deom import Bath
from pyqed.deom import single_oscillator as so
from pyqed.deom import decompose_spectrum_pade as pade
from pyqed.deom import decompose_spectrum_prony as prony
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from pyqed import wavenumber2hartree, au2k, au2wn, au2fs


npsd = 1
lmax = 6
nmod = 3
nsys = 4

temp = 300 / au2k
beta = 1 / temp

omega = 1230 / au2wn
ene = 18950 / au2wn
coupling = 870 / au2wn
lambda_ = 834 / au2wn

hams = np.zeros((nsys, nsys), np.complex128)
hams[1, 1] = ene
hams[2, 2] = ene
hams[1, 2] = coupling
hams[2, 1] = coupling
hams[3, 3] = 2 * ene

qmds = np.zeros((nmod, nsys, nsys), np.complex128)
qmds[0, 1, 1] = 1
qmds[0, 3, 3] = 1
qmds[1, 2, 2] = 1
qmds[1, 3, 3] = 1
qmds[2, 1, 1] = 1
qmds[2, 2, 2] = 1
qmds[2, 3, 3] = 2

sdip = np.zeros((nsys, nsys), np.complex128)
# mode 1
sdip[0, 1] = 1
sdip[2, 3] = 1
sdip[1, 0] = 1
sdip[3, 2] = 1

# mode 2
sdip[0, 2] = np.cos(15/180*np.pi)
sdip[1, 3] = np.cos(15/180*np.pi)
sdip[2, 0] = np.cos(15/180*np.pi)
sdip[3, 1] = np.cos(15/180*np.pi)

sdipu = np.triu(sdip)
sdipl = np.tril(sdip)

print(sdipu, sdipl)

rho = np.zeros((nsys, nsys), dtype=np.complex128)
rho[0, 0] = 1

zeros = np.zeros((nsys, nsys), np.complex128)
gams1, lamd1, gams2, lamd2 = 100 / au2wn, 150 / au2wn, 300 / au2wn, 800 / au2wn
w_sp, lamd_sp, gams_sp, omgs_sp, beta_sp = sp.symbols(
    r"\omega , \lambda, \gamma, \Omega_{s}, \beta", real=True)
sp_vib_para_dict = {lamd_sp: lamd1, gams_sp: gams1, omgs_sp: omega}
sp_el_para_dict = {lamd_sp: lamd2, gams_sp: gams2}
spe_vib_sp = (np.sqrt(2) * lambda_ * omgs_sp / (omgs_sp * omgs_sp - w_sp * w_sp - sp.I *
                                                w_sp * (lamd_sp * omgs_sp / (gams_sp - sp.I * w_sp)))).subs(sp_vib_para_dict)
spe_el_sp = (lamd_sp * gams_sp / (gams_sp - sp.I * w_sp)).subs(sp_el_para_dict)

bath = Bath([spe_vib_sp, spe_vib_sp, spe_el_sp], w_sp, [beta, beta, beta], [2, 2, 1], [
            0, 0, 1, 1, 2, 2], [prony, prony, pade])
mol = Mol(hams, zeros)
deom_solver = mol.deom(bath, [qmds[0], qmds[1], qmds[2]])

deom_solver.set_hierarchy(5)
w = np.linspace(16500/au2wn, 20500/au2wn, 50)
w1 = np.linspace(-20500/au2wn, -16500/au2wn, 50)

# c_w_1 = deom_solver.correlation_4op_3t(
#     sdip, sdip, sdip, sdip, rho, 0, w, w1, if_full=True, if_load=True, lcr='llll')
# np.savez("c_w_1.npz", c_w_1=c_w_1, w=w*au2wn, w1=w*au2wn)

c_w_1 = deom_solver.correlation_4op_3t(
    sdipu, sdipl, sdipl, sdipu, rho, 0/au2fs, w, w1, if_full=True, cut_off_min=0.75, cut_off_max=1.1, if_load=True, lcr='rlrl')
np.savez("c_w_1_0.npz", c_w_1=c_w_1, w=w*au2wn, w1=w1*au2wn)

c_w_2 = deom_solver.correlation_4op_3t(
    sdipu, sdipl, sdipl, sdipu, rho, 0/au2fs, w, w1, if_full=True, cut_off_min=0.75, cut_off_max=1.1, if_load=True, lcr='rrll')
np.savez("c_w_2_0.npz", c_w_1=c_w_2, w=w*au2wn, w1=w1*au2wn)

c_w_3 = deom_solver.correlation_4op_3t(
    sdipu, sdipl, sdipl, sdipu, rho, 0/au2fs, w, w1, if_full=True, cut_off_min=0.75, cut_off_max=1.1, if_load=True, lcr='rlll')
np.savez("c_w_3_0.npz", c_w_1=c_w_3, w=w*au2wn, w1=w1*au2wn)

c_w_1 = deom_solver.correlation_4op_3t(
    sdipu, sdipl, sdipl, sdipu, rho, 10/au2fs, w, w1, if_full=True, cut_off_min=0.75, cut_off_max=1.1, if_load=True, lcr='rlrl')
np.savez("c_w_1_10.npz", c_w_1=c_w_1, w=w*au2wn, w1=w1*au2wn)

c_w_2 = deom_solver.correlation_4op_3t(
    sdipu, sdipl, sdipl, sdipu, rho, 10/au2fs, w, w1, if_full=True, cut_off_min=0.75, cut_off_max=1.1, if_load=True, lcr='rrll')
np.savez("c_w_2_10.npz", c_w_1=c_w_2, w=w*au2wn, w1=w1*au2wn)

c_w_3 = deom_solver.correlation_4op_3t(
    sdipu, sdipl, sdipl, sdipu, rho, 10/au2fs, w, w1, if_full=True, cut_off_min=0.75, cut_off_max=1.1, if_load=True, lcr='rlll')
np.savez("c_w_3_10.npz", c_w_1=c_w_3, w=w*au2wn, w1=w1*au2wn)

c_w_1 = deom_solver.correlation_4op_3t(
    sdipu, sdipl, sdipl, sdipu, rho, 25/au2fs, w, w1, if_full=True, cut_off_min=0.75, cut_off_max=1.1, if_load=True, lcr='rlrl')
np.savez("c_w_1_25.npz", c_w_1=c_w_1, w=w*au2wn, w1=w1*au2wn)

c_w_2 = deom_solver.correlation_4op_3t(
    sdipu, sdipl, sdipl, sdipu, rho, 25/au2fs, w, w1, if_full=True, cut_off_min=0.75, cut_off_max=1.1, if_load=True, lcr='rrll')
np.savez("c_w_2_25.npz", c_w_1=c_w_2, w=w*au2wn, w1=w1*au2wn)

c_w_3 = deom_solver.correlation_4op_3t(
    sdipu, sdipl, sdipl, sdipu, rho, 25/au2fs, w, w1, if_full=True, cut_off_min=0.75, cut_off_max=1.1, if_load=True, lcr='rlll')
np.savez("c_w_3_25.npz", c_w_1=c_w_3, w=w*au2wn, w1=w1*au2wn)


# np.save("c_w_3.npy", c_w_3)
# plt.contourf(w*au2wn, w*au2wn, np.real(c_w_1+c_w_2+c_w_3))
# plt.savefig("2d-spe.png")
# plt.clf()
