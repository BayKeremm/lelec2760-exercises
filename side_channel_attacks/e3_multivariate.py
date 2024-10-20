import numpy as np
from utils_ps3 import load_npz
from e1 import preprocess_traces, sbox
from e2 import pearson_corr, hamw
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_normal

def cov(x, y):
    # Find the covariance between two 1D lists (x and y).
    # Note that var(x) = cov(x, x)
    return np.cov(x, y)[0][1]


def training_phase_cov(index, time_idx, pts, ks, trs):
    """
    index: index of the byte on which to perform the attack.
    time_idx: time sample to consider when building the model
    pts: the plaintexts used in the training set
    ks: the keys used in the training set
    trs: the traces of the training set
    return: the list [us,ss] where
        us: is a numpy array containing the mean of each class
        ss: is a numpy array containing the standard deviation of each class
    """

    means = [np.zeros(len(time_idx)) for _ in range(256) ]
    stds = [np.zeros((len(time_idx), len(time_idx))) for _ in range(256) ]

    nb_traces = trs.shape[0]
    for s_box_out in range(256):
        traces = []
        for key_idx in range(nb_traces):
            key = ks[key_idx,:][index]
            pt = pts[key_idx,:][index]
            if s_box_out == sbox[key ^ pt]:
                traces.append(trs[key_idx,:])

        traces = np.asarray(traces)
        for i in range(len(time_idx)):
            mu_i = np.mean(traces,axis=0)[time_idx[i]]
            means[s_box_out][i] = mu_i
            for j in range(len(time_idx)):
                x = traces[:,time_idx[i]]
                y = traces[:,time_idx[j]]
                stds[s_box_out][i][j] = cov(x,y)


    return [means, stds]
def online_phase_cov(index, time_idx, models, atck_pts, atck_trs):
    """
    index: index of the byte on which to perform the attack.
    time_idx: time sample to consider when building the model
    models: the list [us, ss] corresponding to the models of each class
    atck_pts: plaintexts used in the attack set
    atck_trs: traces of the attack set
    return: a numpy array containing the probability of each byte value.

    assume covariance is equal to zero
    if you increase num of dimensions => more traces to estimate covariance

    """

    nb_traces = atck_trs.shape[0]
    nb_samples = atck_trs.shape[1]
    probabilities = np.zeros(256)
    for i in range(nb_traces):
        l = atck_trs[i,:][time_idx]
        pt = atck_pts[i,:][index]
        for key_guess in range(256):
            s_box_out = sbox[pt ^ key_guess]
            mu_i = models[0][s_box_out] # vector of size (x,)
            covariance_matrix = models[1][s_box_out]
            # create multivariate_normal distribution covariances are assumed to be zero
            d = multivariate_normal(mean=mu_i, cov=covariance_matrix)
            probabilities[key_guess] += np.log(d.pdf(l))

    return probabilities 
def training_phase(index, time_idx, pts, ks, trs):
    """
    index: index of the byte on which to perform the attack.
    time_idx: time sample to consider when building the model
    pts: the plaintexts used in the training set
    ks: the keys used in the training set
    trs: the traces of the training set
    return: the list [us,ss] where
        us: is a numpy array containing the mean of each class
        ss: is a numpy array containing the standard deviation of each class
    """

    means = [np.zeros(len(time_idx)) for _ in range(256) ]
    stds = [np.zeros(len(time_idx)) for _ in range(256) ]

    nb_traces = trs.shape[0]
    for s_box_out in range(256):
        traces = []
        for key_idx in range(nb_traces):
            key = ks[key_idx,:][index]
            pt = pts[key_idx,:][index]
            if s_box_out == sbox[key ^ pt]:
                traces.append(trs[key_idx,:])

        traces = np.asarray(traces)
        for i in range(len(time_idx)):
            mu_i = np.mean(traces,axis=0)[time_idx[i]]
            sigma_i = np.std(traces, axis=0)[time_idx[i]]
            means[s_box_out][i] = mu_i
            stds[s_box_out][i] = sigma_i
    return [means, stds]

def online_phase(index, time_idx, models, atck_pts, atck_trs):
    """
    index: index of the byte on which to perform the attack.
    time_idx: time sample to consider when building the model
    models: the list [us, ss] corresponding to the models of each class
    atck_pts: plaintexts used in the attack set
    atck_trs: traces of the attack set
    return: a numpy array containing the probability of each byte value.

    assume covariance is equal to zero
    if you increase num of dimensions => more traces to estimate covariance

    """

    nb_traces = atck_trs.shape[0]
    nb_samples = atck_trs.shape[1]
    probabilities = np.zeros(256)
    for i in range(nb_traces):
        l = atck_trs[i,:][time_idx]
        pt = atck_pts[i,:][index]
        for key_guess in range(256):
            s_box_out = sbox[pt ^ key_guess]
            mu_i = models[0][s_box_out] # vector of size (x,)
            sigma_i = models[1][s_box_out]
            # create multivariate_normal distribution covariances are assumed to be zero
            covariance_matrix = np.diag(sigma_i ** 2)
            d = multivariate_normal(mean=mu_i, cov=covariance_matrix)
            probabilities[key_guess] += np.log(d.pdf(l))

    return probabilities 


def ta_byte(index, train_pts, train_ks, train_trs, atck_pts, atck_trs):
    """
    index: index of the byte on which to perform the attack.
    train_pts: the plaintexts used in the training set
    train_ks: the keys used in the training set
    train_trs: the traces of the training set
    atck_pts: plaintexts used in the attack set
    atck_trs: traces of the attack set
    return: a np.array with the key bytes, from highest probable to less probable
    """
    numberPOIs = 2
    nb_traces =  train_trs.shape[0]
    nb_samples = train_trs.shape[1]
    weights = []

    for i in range(nb_traces):
        x_i = train_pts[i,:][index]
        k_i = train_ks[i,:][index]
        v_i = sbox[x_i ^ k_i] 
        hw = hamw(v_i)
        weights.append(np.full(nb_samples,hw))

    p_corr = pearson_corr(weights,train_trs)

    sorted_indices = np.argsort(np.abs(p_corr))
    largest_indexes = sorted_indices[-numberPOIs:]

    models = training_phase(index, largest_indexes, train_pts, train_ks, train_trs)
    probabilities = online_phase(index, largest_indexes, models, atck_pts, atck_trs)

    return np.array(sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True))


def run_full_ta_known_key(
    tr_plain, tr_keys, tr_trs, atck_plain, atck_key, atck_trs, idx_bytes
):
    print("Run TA the with the known key for bytes idx: \n{}".format(idx_bytes))
    key_bytes_found = 16 * [0]
    key_ranks = 16 * [0]
    for i in idx_bytes:
        print("Run TA for byte {}...".format(i), flush=True)
        ta_res = ta_byte(i, tr_plain, tr_keys, tr_trs, atck_plain, atck_trs)
        key_bytes_found[i] = ta_res[0]
        key_ranks[i] = np.where(ta_res == atck_key[i])[0]
        print("Rank of correct key: {}".format(key_ranks[i]), end="")
        if key_bytes_found[i] == atck_key[i]:
            print("")
        else:
            print("--> FAILURE")

    hits = np.sum(atck_key[:] == key_bytes_found)
    print("{} out of {} correct key bytes found.".format(hits, len(idx_bytes)))
    print("Avg key rank: {}".format(np.mean(key_ranks)))
    return key_ranks


if __name__ == "__main__":
    # Load the data
    train_dataset = load_npz("training_set.npz")
    train_plaintexts = train_dataset["xbyte"]
    train_keys = train_dataset["kv"]
    train_traces = train_dataset["traces"].astype(float)

    am_train_tr = min(10000, train_plaintexts.shape[0])
    train_plaintexts = train_plaintexts[:am_train_tr, :]
    train_keys = train_keys[:am_train_tr, :]
    train_traces = train_traces[:am_train_tr, :]

    #
    atck_dataset = load_npz("attack_set_known_key.npz")
    atck_plaintexts = atck_dataset["xbyte"]
    atck_key = atck_dataset["kv"][0, :]
    atck_traces = atck_dataset["traces"].astype(float)

    am_atck_tr = min(40, atck_traces.shape[0])
    atck_plaintexts = atck_plaintexts[:am_atck_tr, :]
    atck_traces = atck_traces[:am_atck_tr, :]

    # Preprocess traces
    train_traces = preprocess_traces(train_traces[:,2600:3900])
    atck_traces = preprocess_traces(atck_traces[:,2600:3900])

    # Run the attack
    # Indexes of byte to attack
    idxes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    run_full_ta_known_key(
        train_plaintexts,
        train_keys,
        train_traces,
        atck_plaintexts,
        atck_key,
        atck_traces,
        idxes,
    )
