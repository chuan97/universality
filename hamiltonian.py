import numpy as np
from scipy.linalg import kron

import spin

# Classes for families of Hamiltonians (good idea!)
    
class SpinMonomer:
    def __init__(self, S, g, h=None, A=None, shift=None):
        self.S = S
        self.g = g
        self.h = h
        self.A = A
        self.shift = shift
        
        Sz, Sp, Sm, self.Seye = spin.spin_operators(self.S, to_dense_array=True, dtype=complex)
        Sx = 1/2 * (Sp + Sm)
        Sy = -1j/2 * (Sp - Sm)
        self.Sarray = np.array([Sx, Sy, Sz])
    
    @property
    def H(self):
        H = np.zeros(self.Seye.shape, dtype=complex)
        
        if self.h is not None:
            H += self.external_field(self.h)
        if self.A is not None:
            H += np.sum(self.Sarray @ np.tensordot(self.A, self.Sarray, axes=[1, 0]), axis=0)
        if self.shift:
            H += self.shift * self.Seye
            
        return H
        
    def external_field(self, h_ext):
        Hext = np.tensordot(h_ext, np.tensordot(self.g, self.Sarray, axes=[1, 0]), axes=[0, 0])
        return Hext

class SpinDimer:
    def __init__(self, Ss, gs, J, hs=None, As=None, shift=None):
        self.Ss = Ss
        self.gs = gs
        self.J = J
        self.hs = hs
        self.As = As
        self.shift = shift
        
        self.Sarrays = []
        self.Seyes = []
        for i, S in enumerate(Ss):
            Sz, Sp, Sm, Seye = spin.spin_operators(S, to_dense_array=True, dtype=complex)
            Sx = 1/2 * (Sp + Sm)
            Sy = -1j/2 * (Sp - Sm)
            self.Sarrays.append(np.array([Sx, Sy, Sz]))
            self.Seyes.append(Seye)
            
        self.Sarrays[0] = np.array([np.kron(comp, self.Seyes[1]) for comp in self.Sarrays[0]])
        self.Sarrays[1] = np.array([np.kron(self.Seyes[0], comp) for comp in self.Sarrays[1]])
        
    @property
    def H(self):
        H = np.zeros(self.Sarrays[0][0].shape, dtype=complex)
        
        if self.hs is not None:
            H += self.external_field(self.hs)
        if self.J is not None:
            H += np.sum(self.Sarrays[0] @ np.tensordot(self.J, self.Sarrays[1], axes=[1, 0]), axis=0)
        if self.As is not None:
            for A, Sarray in zip(self.As, self.Sarrays):
                H += np.sum(Sarray @ np.tensordot(A, Sarray, axes=[1, 0]), axis=0)
        if self.shift:
            H += self.shift * self.Seye
            
        return H 
    
    def external_field(self, h_exts):
        Hext = np.zeros(self.Sarrays[0][0].shape, dtype=complex)
        
        for h_ext, g, Sarray in zip(h_exts, self.gs, self.Sarrays):
            Hext += np.tensordot(h_ext, np.tensordot(g, Sarray, axes=[1, 0]), axes=[0, 0])
            
        return Hext

# hard coded Hamiltonians and external fields (bad idea!)

def VOporphirin(A, g, h):  
    S = 1/2
    Sz, Sp, Sm, Seye = spin.spin_operators(S, to_dense_array=True, dtype=complex)
    Sx = 1/2 * (Sp + Sm)
    Sy= -1j/2 * (Sp - Sm)
    
    I = 7/2
    Iz, Ip, Im, Ieye = spin.spin_operators(I, to_dense_array=True, dtype=complex)
    Ix = 1/2 * (Ip + Im)
    Iy= -1j/2 * (Ip - Im)
    
    HS = A[0]*kron(Ix, Sx) + A[1]*kron(Iy, Sy) + A[2]*kron(Iz, Sz)
    HS += h[0]*g[0]*kron(Ieye, Sx) + h[1]*g[1]*kron(Ieye, Sy) + h[2]*g[2]*kron(Ieye, Sz)
    return HS

def external_field_VOporphirin(h_ext, g):
    S = 1/2
    Sz, Sp, Sm, Seye = spin.spin_operators(S, to_dense_array=True, dtype=complex)
    Sx = 1/2 * (Sp + Sm)
    Sy= -1j/2 * (Sp - Sm)
    
    I = 7/2
    Iz, Ip, Im, Ieye = spin.spin_operators(I, to_dense_array=True, dtype=complex)
    
    Hext = h_ext[0]*g[0]*kron(Ieye, Sx) + h_ext[1]*g[1]*kron(Ieye, Sy) + h_ext[2]*g[2]*kron(Ieye, Sz)
    return Hext

def GdW30(D1, E1, gmub, hx, hy, hz):
    S = 7/2
    Sz, Sp, Sm, Seye = spin.spin_operators(S, to_dense_array=True, dtype=complex)
    Sx = 1/2 * (Sp + Sm)
    Sy= -1j/2 * (Sp - Sm)
    
    HS = D1*Sz@Sz + E1*(Sx@Sx - Sy@Sy)
    HS += -D1 * 1/3 * S*(S + 1) * Seye
    HS += -gmub * (hx*Sx + hy*Sy + hz*Sz)
    return HS

def external_field_GdW30(θ, ϕ):
    S = 7/2
    Sz, Sp, Sm, Seye = spin.spin_operators(S, to_dense_array=True, dtype=complex)
    Sx = 1/2 * (Sp + Sm)
    Sy= -1j/2 * (Sp - Sm)
    
    Hext = -np.cos(θ)*Sz - np.sin(θ)*(np.cos(ϕ)*Sx + np.sin(ϕ)*Sy)
    return Hext