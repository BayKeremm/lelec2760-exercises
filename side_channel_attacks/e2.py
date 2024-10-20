import numpy as np
from utils_ps3 import load_npz
from e1 import  preprocess_traces#,AK_SB_1byte
import matplotlib.pyplot as plt
import tqdm


sbox = load_npz("AES_Sbox.npz")["AES_Sbox"][0]

def hamw(v):
    c = 0
    while v:
        c += 1
        v &= v - 1

    return c

def pearson_corr(x, y):
    """
    x: raw traces, as an np.array of shape (nb_traces, nb_samples)
    y: model, as an np.array of shape (nb_traces, nb_samples)
    return: the pearson coefficient of each sample as a np.array of shape (1, nb_samples)
    """
    
    #TODO:
    x = np.asarray(x)
    n = x.shape[0]
    nb_samples = x.shape[1]
    corr_coeffs = np.zeros(nb_samples)

    for j in range(nb_samples):
        x_j = x[:,j]
        y_j = y[:,j]
        x_sum = np.sum(x_j)
        y_sum = np.sum(y_j)
        x2_sum = np.sum(x_j**2)
        y2_sum = np.sum(y_j**2)

        numerator = n * np.matmul(np.transpose(x_j), y_j) - x_sum * y_sum
        denominator = np.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))
        corr_coeffs[j] = numerator / denominator

    return corr_coeffs


def cpa_byte_out_sbox(index, pts, traces):
    """
    index: index of the byte on which to perform the attack.
    pts: the plaintext of each encryption performed.
    traces: the power measurements performed for each encryption.
    return: an np.array with the key bytes, from highest probable to less probable
    when performing the attack targeting the input of the sbox
    """
    # TODO
    # compute the corr coef, put it into an array, and then rank
    # you get the hamming weight, extend that as an array to the number of samples and compute the corr coefficient with that
    nb_traces = traces.shape[0]
    nb_samples = traces.shape[1]
    coeffs = []
    for key_guess in range(256):
        weights = []
        for i in range(nb_traces):
            x_i = pts[i,:][index] # get the byte of the plaintext
            v_i_star = sbox[x_i ^ key_guess] # apply the prediction
            hw = hamw(v_i_star)
            weights.append(np.full(nb_samples,hw))
        p_corr = pearson_corr(weights,traces)
        coeffs.append(np.max(abs(p_corr)))

    return np.array(sorted(range(len(coeffs)), key=lambda i: coeffs[i], reverse=True))


'''
linear model on a linear operation
with the sbox the hamming weight is not related at all
thanks to the nonlinearity, the modelling overlaps less 

for dpa attacks sboxes are problem
non-linearity allows larger attack surface
'''

def cpa_byte_in_sbox(index, pts, traces):
    """
    index: index of the byte on which to perform the attack.
    pts: the plaintext of each encryption performed.
    traces: the power measurements performed for each encryption.
    return: an np.array with the key bytes, from highest probable to less probable
    when performing the attack targeting the output of the sbox
    """
    # TODO
    nb_traces = traces.shape[0]
    nb_samples = traces.shape[1]
    coeffs = []
    for key_guess in range(256):
        weights = []
        for i in range(nb_traces):
            x_i = pts[i,:][index] # get the byte of the plaintext
            v_i_star = x_i ^ key_guess # apply the prediction
            hw = hamw(v_i_star)
            weights.append(np.full(nb_samples,hw))
        p_corr = pearson_corr(weights,traces)
        coeffs.append(np.max(abs(p_corr)))

    return np.array(sorted(range(len(coeffs)), key=lambda i: coeffs[i], reverse=True))
    


def run_full_cpa_known_key(pts, ks, trs, idx_bytes, out=True):
    print("Run CPA the with the known key for bytes idx: \n{}".format(idx_bytes))
    if out:
        print("Target: sbox output")
    else:
        print("Target: sbox input")
    key_bytes_found = 16 * [0]
    key_ranks = 16 * [0]
    for i in idx_bytes:
        print("Run CPA for byte {}...".format(i), flush=True)
        if out:
            cpa_res = cpa_byte_out_sbox(i, pts, trs)
        else:
            cpa_res = cpa_byte_in_sbox(i, pts, trs)
        key_bytes_found[i] = cpa_res[0]
        key_ranks[i] = np.where(cpa_res == ks[0, i])[0]
        print("Rank of correct key: {}".format(key_ranks[i]), end="")
        if key_bytes_found[i] == ks[0, i]:
            print("")
        else:
            print("--> FAILURE")

    hits = np.sum(ks[0, :] == key_bytes_found)
    print("{} out of {} correct key bytes found.".format(hits, len(idx_bytes)))
    print("Avg key rank: {}".format(np.mean(key_ranks)))
    return key_ranks


if __name__ == "__main__":
    # Load the data
    dataset = load_npz("attack_set_known_key.npz")
    plaintexts = dataset["xbyte"]
    keys = dataset["kv"]
    traces = dataset["traces"].astype(float)

    # Amount trace taken
    am_tr = min(800, plaintexts.shape[0])

    plaintexts = plaintexts[:am_tr, :]
    print(plaintexts.shape)
    keys = keys[:am_tr, :]
    traces = traces[:am_tr, :]

    #pearson_corr(traces,traces)

    # Prepocess traces
    traces = preprocess_traces(traces[:,2700:3900])

    # Run the attack
    # Indexes of byte to attack
    idxes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    run_full_cpa_known_key(plaintexts, keys, traces, idxes, out=True)
    run_full_cpa_known_key(plaintexts, keys, traces, idxes, out=False)
