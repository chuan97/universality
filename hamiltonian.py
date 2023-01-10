import numpy as np
from scipy.linalg import kron

import spin

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

def external_field(h_ext, g):
    S = 1/2
    Sz, Sp, Sm, Seye = spin.spin_operators(S, to_dense_array=True, dtype=complex)
    Sx = 1/2 * (Sp + Sm)
    Sy= -1j/2 * (Sp - Sm)
    
    I = 7/2
    Iz, Ip, Im, Ieye = spin.spin_operators(I, to_dense_array=True, dtype=complex)
    
    Hext = h_ext[0]*g[0]*kron(Ieye, Sx) + h_ext[1]*g[1]*kron(Ieye, Sy) + h_ext[2]*g[2]*kron(Ieye, Sz)
    return Hext

class _VOporphirin:
    def __init__(self, A, g, h, S, I):
        self.A = A
        self.g = g
        self.h = h
        self.S = S
        self.I = I
        
        Sz, Sp, Sm, self.Seye = spin.spin_operators(self.S, to_dense_array=True, dtype=complex)
        Sx = 1/2 * (Sp + Sm)
        Sy= -1j/2 * (Sp - Sm)
        self.Sarray = np.array([Sx, Sy, Sz])
        
        Iz, Ip, Im, self.Ieye = spin.spin_operators(self.I, to_dense_array=True, dtype=complex)
        Ix = 1/2 * (Ip + Im)
        Iy= -1j/2 * (Ip - Im)
        self.Iarray = np.array([Ix, Iy, Iz])
        
    @property
    def H(self):
        HS = np.tensordot(self.Iarray, np.tensordot(self.A, self.Sarray, axes=[1, 0]), axes=[0, 0])
        print(HS.shape)
        HS += self.external_field(self.h)
        return HS
    
    def external_field(self, h_ext):
        Hext = np.tensordot(h_ext, np.tensordot(self.g, self.Sarray, axes=[1, 0]), axes=[0, 0])
        Hext = kron(self.Ieye, Hext)
        print(Hext.shape)
        return Hext
    