import numpy as np
from e1 import encrypt
from utils import dec2bits

def gen_data(k,rounds,n):
    """
        Generate plaintext - ciphertext pairs for different attacks
    """
    pts = np.zeros((n,8),dtype=np.uint8)
    cts = np.zeros((n,8),dtype=np.uint8)
    
    # TODO
    for i in range(n):
        r = np.random.randint(0,16,8,dtype=np.uint8)
        pts[i] = r
        cts[i]=encrypt(r,k,rounds)
    
    return pts,cts

def experimental_bias(pts,pts_msk,cts,cts_msk):
    """
        pts: plaintexts, decimal form
        pts_msk: input mask, decimal form 
        cts: ciphertext, decimal form
        cts_msk: output mask, decimal form

        compute experimental
    """
    n = len(pts)
    
    # TODO
    counter = 0
    for i in range(n):
        pt = pts[i]
        ct = cts[i]
        pt_masked = pt & pts_msk
        ct_masked = ct & cts_msk

        pt_xor = 0
        for pt_ in pt_masked:
            pt_bin = bin(pt_)[2:]
            for b in pt_bin:
                pt_xor ^= int(b)

        ct_xor = 0
        for ct_ in ct_masked:
            ct_bin = bin(ct_)[2:]
            for b in ct_bin:
                ct_xor ^= int(b)


        if pt_xor^ct_xor==0:
            counter +=1

    return float((counter -n/2) / n)


if __name__ == "__main__":
    
    # example masks for chained Sbox. 
    #pts_msk = [0x9, 0, 0, 0, 0, 0, 0, 0]
    #cts_msk = [0, 0x2, 0X0, 0x2, 0, 0x2, 0, 0x2]
    pts_msk = [0x8, 0, 0, 0, 0, 0, 0, 0]
    cts_msk = [0x8, 0, 0x8, 0, 0, 0, 0, 0]
    k = np.random.randint(0,16,8,dtype=np.uint8)
    n = 3000
    rounds = 2

    pts,cts = gen_data(k,2,n)
    bias = experimental_bias(pts,pts_msk,cts,cts_msk)
    print("bias:", bias)
    print("Pr:", bias + .5)
