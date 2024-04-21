import numpy as np
from utils_ps3 import load_npz, plot_traces, dec2bits

sbox = load_npz("AES_Sbox.npz")["AES_Sbox"][0]


def preprocess_traces(traces):
    """
    traces: raw traces.
    return: return the preprocessed set of traces
    """
    #return traces

    ret = traces.copy()
    for i in range(traces.shape[0]):
        #plot_traces(traces[i,:])
        ret[i, :] -= np.mean(traces[i, :])
        #plot_traces(ret[i,:])
    return ret


def dpa_byte(index, pts, traces):
    """
    index: index of the byte on which to perform the attack.
    pts: the plaintext of each encryption performed.
    traces: the power measurements performed for each encryption.
    return: an np.array with the key bytes, from highest probable to less probable
    """

    # TODO
    mean_diffs  = []
    for key_guess in range(256):
        set0 = []
        set1 = []
        for i in range(traces.shape[0]):
            x_i = pts[i,:][index] # get the byte of the plaintext
            v_i_star = sbox[x_i ^ key_guess] # apply the prediction
            trace = traces[i,:] # (16000,)
            if v_i_star % 2 == 1: # If LSB = 1 => modulo 2 returns 1
                set1.append(trace)
            else:
                set0.append(trace)
        set0_avg = np.asarray(set0).mean(axis=0)
        set1_avg = np.asarray(set1).mean(axis=0)
        mean_diffs.append(np.max(abs(set0_avg-set1_avg)))

    # sort i from 0 to 255 with key mean_diffs[i]
    return np.array(sorted(range(len(mean_diffs)), key=lambda i: mean_diffs[i], reverse=True))


def run_full_dpa_known_key(pts, ks, trs, idx_bytes):
    print("Run DPA the with the known key for bytes idx: \n{}".format(idx_bytes))

    key_bytes_found = 16 * [0]
    key_ranks = 16 * [0]
    for i in idx_bytes:
        print("Run DPA for byte {}...".format(i), end="", flush=True)
        dpa_res = dpa_byte(i, pts, trs)
        key_bytes_found[i] = dpa_res[0]
        key_ranks[i] = np.where(dpa_res == ks[0, i])[0]
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
    am_tr = min(1200, plaintexts.shape[0])

    plaintexts = plaintexts[:am_tr, :]
    keys = keys[:am_tr, :]
    traces = traces[:am_tr, :]

    # Uncomment the next line to plot the first trace
    #plot_traces(traces[0,:])

    # Preprocess traces
    traces = preprocess_traces(traces[:,2600:3900])

    # Run the attack
    # Indexes of byte to attack
    idxes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    run_full_dpa_known_key(plaintexts, keys, traces, idxes)
