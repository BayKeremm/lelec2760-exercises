import numpy as np
from utils import dec2bits, bits2dec

sbox = np.load("sbox.npz")["sbox"]
sboxinv = np.load("sbox.npz")["sboxinv"]

player = np.load("player.npz")["player"]
playerinv = np.load("player.npz")["playerinv"]


def encrypt(pt, k, nrounds):
    """
        pt: plaintext
        k: key
        nrounds: number of rounds
        return: the ciphertext as a np.array
    #print("S-box: ", sbox)
    #print("Player : ", player)
        #print("Round: ",i, "p_i_bits: ", p_i_bits)
        #print("Round: ",i, "k_bits:   ", k_bits)
        print("Round: ",i, "u_i_bits: ", u_i_bits)
        print("Round: ",i, "u_i: ", u_i)
        print("Round: ",i, "v_i: ", v_i)
        print("Round: ",i, "v_i_end: ", v_i_end)
    print("Round: ",i, "p_i before last key: ", p_i)
    """

    # TODO
    p_i = pt
    k_bits = dec2bits(k)
    for i in range(nrounds):
        p_i_bits = dec2bits(p_i)
        u_i_bits = p_i_bits ^ k_bits
        u_i = bits2dec(u_i_bits)
        v_i = u_i.copy()
        for j in range(len(u_i)):
            #S-box
            v_i[j] = sbox[u_i[j]]
        #permutation
        v_i_bits = dec2bits(v_i)
        v_i_end = bits2dec(v_i_bits[player]) 
        p_i = v_i_end
    #last round key
    ct = bits2dec(dec2bits(p_i) ^ k_bits)
    
    return ct


def decrypt(ct, k, nrounds):
    """
        ct: ciphertext 
        k: key
        nrounds: number of rounds
        return: the plaintext as a np.array
    """
    # TODO
    k_bits = dec2bits(k)
    #get the ct before last key xor
    v_i = bits2dec(dec2bits(ct) ^ k_bits)

    for i in range(0 , nrounds):
        # inv permute
        v_i = bits2dec(dec2bits(v_i)[playerinv])
        # inv s-box
        u_i = v_i.copy()

        u_i = sboxinv[v_i]
        
        #undo key
        v_i = bits2dec(dec2bits(u_i) ^ k_bits)

    return v_i


if __name__ == "__main__":
    print("LELEC2760: TP 1 - Ex 1")

    print(" -> Run known answer tests")
    key = [0, 1, 2, 3, 4, 5, 6, 7]
    plain = [0, 7, 0, 2, 2, 0, 1, 2]
    ctexts = [
        [12, 11, 9, 1, 2, 13, 7, 5],
        [6, 3, 9, 13, 6, 9, 7, 1],
        [14, 15, 1, 4, 11, 9, 3, 4],
    ]

    for r in range(1, 4):
        assert (ctexts[r - 1] == encrypt(plain, key, r)).all()
        assert (plain == decrypt(ctexts[r - 1], key, r)).all()

    print(" -> Run random tests")

    for _ in range(100):
        k = np.random.randint(0, 16, 8)
        pt = np.random.randint(0, 16, 8)
        assert (decrypt(encrypt(pt, k, 3), k, 3) == pt).all()
