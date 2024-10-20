import numpy as np
from e1 import encrypt,decrypt
from utils import dec2bits,bits2dec
from e3 import gen_data,experimental_bias
from tqdm import tqdm

sboxinv = np.load("sbox.npz")["sboxinv"]
playerinv = np.load("player.npz")["playerinv"]

def attack(pts,pts_msk,cts,cts_msk,key_target):
    key_bits_target = dec2bits(key_target)
    guess_size = int(np.sum(key_bits_target))
    key_bits_locs = np.where(key_bits_target == 1)[0]

    bias = np.zeros(2**guess_size)
    for key_guess in tqdm(range(2**guess_size)):
        k = np.zeros(32,dtype=np.uint8)
        k[key_bits_locs] = dec2bits([key_guess],size=guess_size)

        cts_N_minus_1 = []
        for i in range(len(pts)):
            cts_N_minus_1.append(decrypt(cts[i] ,bits2dec(k),1)  ^ bits2dec(k))

        bias[key_guess] = np.abs(experimental_bias(pts,pts_msk,cts_N_minus_1,cts_msk))

    key_guess = 0
    
    return bias

if __name__ == "__main__":
    
    pts_msk = [0x8, 0, 0, 0, 0, 0, 0, 0]
    cts_msk = [0x8, 0, 0x8, 0, 0, 0, 0, 0]

    key_target = [0xa, 0, 0xa, 0, 0xa, 0, 0xa, 0]
 
    # known key 
    n = 3000
    key = np.random.randint(0,16,8,dtype=np.uint8)
    pts,cts = gen_data(key,3,n)
    
    #TODO attack unknown key
    #pts = np.load("pts_cts_pairs.npz")["pts"][:,:n]
    #cts = np.load("pts_cts_pairs.npz")["cts"][:,:n]
   
    bias = attack(pts,pts_msk,cts,cts_msk,key_target)

    # evaluation
    key_bits = dec2bits(key)
    key_bits_target = dec2bits(key_target)
    guess_size = int(np.sum(key_bits_target))
    key_bits_locs = np.where(key_bits_target == 1)[0]

    real_k = bits2dec(key_bits[key_bits_locs],size=guess_size)
    print("real key word:",real_k)
    print("key guess:",np.argmax(np.abs(bias)))
    print("location:",(2**guess_size) - np.where(np.argsort(np.abs(bias))==real_k)[0])

            #ct_xor = 0
            #for ct_ in u_N_masked: 
                #ct_bin = bin(ct_)[2:]
                #for b in ct_bin:
                    #ct_xor ^= int(b)
            
            #pt_masked = pts[i] & pts_msk
            #pt_xor = 0
            #for pt_ in pt_masked:
                #pt_bin = bin(pt_)[2:]
                #for b in pt_bin:
                    #pt_xor ^= int(b)

            #if ct_xor ^pt_xor == 0:
                #count +=1

        #bias[key_guess] = count / len(pts)