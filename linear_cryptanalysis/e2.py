import numpy as np
from utils import dec2bits, bits2dec


def compute_bias_table(sbox, l):
    """

    """
    
    # first dimension in table is input mask
    # second dimension in table is output mask
    table = np.zeros((l, l))

    n = np.log2(l)
    #TODO
    for i in range(l):
        for j in range(l):
            counter = 0
            for k in range(l):
                X = k
                Y = sbox[X]

                #mask inputs
                X_masked = i & X
                Y_masked = j & Y

                X_bin = bin(X_masked)[2:]
                Y_bin = bin(Y_masked)[2:]

                X_xor = 0
                for bit in X_bin:
                    X_xor ^= int(bit)

                Y_xor = 0
                for bit in Y_bin:
                    Y_xor ^= int(bit)

                if X_xor ^ Y_xor == 0:
                    counter = counter + 1

            table[i][j] = (counter - l/2)/16
            counter = 0
    return table


def print_bias_table(table):
    l = len(table)
    print("    |" + "|".join(["%8x" % (x) for x in range(l)]))
    for i in range(l):
        print("%4x|" % (i) + "|".join([" %6.3f " % (x) for x in table[i, :]]))


if __name__ == "__main__":

    sbox = np.load("sbox.npz")["sbox"]
    bias_table = compute_bias_table(sbox, 16)

    print_bias_table(bias_table)

    np.save("my_bias_table.npy", bias_table)
