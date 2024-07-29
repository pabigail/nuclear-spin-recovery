import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pycce as pc
import random
from mpl_toolkits.mplot3d import Axes3D


def make_df_from_Ivady_file(file_path, strong_thresh, weak_thresh):
    '''
    input:
        file_path (string): location of text file with hyperfine couplings (MHz) and locations (A) 
        of nuclear spins
        strong_thresh, weak_thresh (floats): kHz thresholds to remove all hyperfine couplings 
        with a component stronger than strong_thresh and with both couplings weaker than weak_thresh
    
    output:
        hf_df: dataframe with hyperfine couplings (kHz) and locations (A) of all
        nuclear spins in files
    '''
    
    hf_data = pd.read_csv(file_path, sep=' ', header=None, names=['distance', # A
                                                              'x', # A
                                                              'y', # A
                                                              'z', # A
                                                              'A_xx', # MHz
                                                              'A_yy', # MHz
                                                              'A_zz', # MHz
                                                              'A_xy', # MHz
                                                              'A_xz', # MHz
                                                              'A_yz']) # MHz
    print(len(hf_data))

    # transform data into KHz and add A_par and A_perp
    MHz_to_KHz = 1000
    hf_data['A_xx'] *= MHz_to_KHz
    hf_data['A_yy'] *= MHz_to_KHz
    hf_data['A_zz'] *= MHz_to_KHz
    hf_data['A_xy'] *= MHz_to_KHz
    hf_data['A_xz'] *= MHz_to_KHz
    hf_data['A_yz'] *= MHz_to_KHz
    hf_data['A_perp'] = np.sqrt(hf_data['A_xz']**2 + hf_data['A_yz']**2)
    hf_data['A_par'] = hf_data['A_zz']

    # all spins must be below strong threshold
    hf_df = hf_data[((hf_data['A_perp'] <= strong_thresh) & 
                     (np.abs(hf_data['A_par']) <= strong_thresh))]
    # one component must be above weak threshold
    hf_df = hf_df[((hf_df['A_perp'] >= weak_thresh) | 
                    (np.abs(hf_df['A_par']) >= weak_thresh))]
    
    # remove all such that A_perp is below 1 kHz
    hf_df = hf_df[(hf_df['A_perp'] >= 1)]
    
    return hf_df


def get_hf_limit(df_data, num_pulses, noise, confidence):
    '''
    input:
        df_data (dataframe) : contains num_pulses, hf_thresh, confidence, and noise_tol columns
        num_pulses (int or string): right now, needs to be 8, 16, or 'joint'
        noise (float): amount of noise from experiment (np.linalg.norm(uncertainty of each data point based on photon noise distribution)
        confidence (float): right now needs to be 0.5, 0.75, 0.9, 0.95, 0.99, or 0.999
    output:
        hf_detect (float): based on bootstrapping, or np.nan if cannot find based on available data
    '''

    sub_df = df_data.loc[((df_data['confidence']==confidence) &\
                    (df_data['num_pulses']==num_pulses)), 
                    ['hf_thresh', 'noise_tol']]
    
    hf = np.array(sub_df['hf_thresh'])
    noise_tol = np.array(sub_df['noise_tol'])
    zeros = np.zeros(np.shape(noise_tol))
    
    try:
        xp = np.linspace(hf[0], hf[-1], 10000)
    except IndexError:
        return np.nan
    interpolated = np.interp(xp, hf, noise_tol)
    
    noise_thresh_line = np.full(np.shape(xp), noise)
    
    sign_array = np.sign(interpolated - noise_thresh_line)
    if sign_array[-1] == -1:
        hf_detect = np.nan
    else:
        try:
            hf_detect_index = np.where((np.diff(np.sign(interpolated - noise_thresh_line)) != 0)*1 == 1)[0][0]
            hf_detect = xp[hf_detect_index]
        except IndexError:
            hf_detect = np.nan
            
    return hf_detect


def calc_distance_two_atoms(hf_df, row_index_1, row_index_2):
    '''
    input:
        hf_df: data frame of hyperfines, locations of spins 
        row_index_1, row_index_2: integers in [0, len(hf_df)]
    output:
        distance (float): L2 distance of cartesian coordinates
        corresponding to nuclear spins corresponding to row indices
    '''
    
    coords_atom_1 = np.array([hf_df.iloc[row_index_1]['x'],
                              hf_df.iloc[row_index_1]['y'],
                              hf_df.iloc[row_index_1]['z']])
    
    coords_atom_2 = np.array([hf_df.iloc[row_index_2]['x'],
                              hf_df.iloc[row_index_2]['y'],
                              hf_df.iloc[row_index_2]['z']])
    
    return np.linalg.norm(coords_atom_1 - coords_atom_2)


def get_distance_matrix(hf_df):
    '''
    input:
        hf_df (dataframe): contains hyperfines, locations of spins
    output:
        dist_matrix (numpy array): a symmetric matrix where each entry 
        [i,j] = [j,i] is the distance between spins i and j in hf_df
    '''

    dist_matrix = np.zeros((len(hf_df), len(hf_df)))
    for i in range(len(hf_df)):
        for j in range(len(hf_df)):
            if i > j:
                distance = calc_distance_two_atoms(hf_df, i, j)
                dist_matrix[i][j] = distance
                dist_matrix[j][i] = distance 
                
    return dist_matrix


def get_specific_exp_parameters(num_pulses):
    '''
    input:
        num_pulses (int): must be 8 or 16
    output:
        BEGIN_TIME (float): ms 
        END_TIME (float): ms
        NUM_TIMEPOINTS (int): number of timepoints
        TIME (1d numpy array): timepoints 
    '''
    
    if num_pulses == 8:
        BEGIN_TIME = 6.0
        END_TIME = 125.52
        NUM_TIMEPOINTS = 250
        TIME = np.linspace(BEGIN_TIME, END_TIME, NUM_TIMEPOINTS)*1e-3/(2*num_pulses)
        return BEGIN_TIME, END_TIME, NUM_TIMEPOINTS, TIME
    
    elif num_pulses == 16:
        BEGIN_TIME = 12.0
        END_TIME = 251.04
        NUM_TIMEPOINTS = 250
        TIME = np.linspace(BEGIN_TIME, END_TIME, NUM_TIMEPOINTS)*1e-3/(2*num_pulses)
        return BEGIN_TIME, END_TIME, NUM_TIMEPOINTS, TIME
    
    else:
        raise ValueError("Pulse number must be 8 or 16")
        
        
def is_array_like(obj):
    '''
    input:
        obj: almost anything
    output:
        boolean if object is a list, tuple or numpy array
    '''
    return isinstance(obj, (list, tuple, np.ndarray))
        
    
def make_exp_params_dict(num_exps, num_pulses, mag_field, noise, timepoints):
    '''
    input:
        num_exps (int): number of different experiments
        num_pulses (array-like of ints): list of integers corresponding to number 
        of pulses in each experiment
        mag_field (array-like of floats): list of magnetic fields (in Gauss) for each
        experiment
        noise (array-like of floats): list of sigma^2 amount of noise in each experiment
        timepoints (array-like of array-like of floats): for each experiment, an array of timepoints
        at which experimental data is collected
    output:
        exp_params (dict): dict with entries "num_experiments" as an int and the rest are 
        lists where each entry corresponds to a single experiment
        
    want to make sure everything is iterable so can input both single experiments or multiple
    experiments; if there is a single experiment input can be list of length one or just the 
    entry and will cast to iterable array
    '''
    exp_params = {}
    exp_params['num_experiments'] = num_exps
    
    if is_array_like(mag_field):
        exp_params['mag_field'] = np.array(mag_field)
    else:
        exp_params['mag_field'] = np.array([mag_field])
    
    if is_array_like(noise):
        exp_params['noise'] = np.array(noise)
    else:
        exp_params['noise'] = np.array([noise])
    
    if is_array_like(num_pulses):
        exp_params['num_pulses'] = np.array(num_pulses)
    else:
        exp_params['num_pulses'] = np.array([num_pulses])
        
    if is_array_like(timepoints[0]):
        exp_params['timepoints'] = np.array(timepoints)
    else:
        exp_params['timepoints'] = np.array([timepoints])
        
    return exp_params


def coherence_one_spin(t_i, A_par, A_perp, N, B_mag):
    '''
    input:
        t_i (array-like of floats): timepoints at which coherence data was taken
        A_par (float): hyperfine coupling for a single spin (kHz)
        A_perp (float): hyperfine coupling for a single spin (kHz)
        N (int): number of pulses
        B_mag (float): magnetic field (Gauss)
    output:
        coherence_signal_one_spin (1d array-like of floats): same dimensions as t_i
    '''
    
    A_par = A_par * 2 * np.pi
    A_perp = A_perp * 2 * np.pi
    
    w_L = pc.common_isotopes['13C'].gyro*B_mag
    
    w_1 = A_par + w_L
    w = np.sqrt((w_1)**2 + A_perp**2)
    mz = w_1 / w
    mx = A_perp / w

    alpha = np.outer(t_i, w)
    beta = np.array(t_i * w_L)[:, np.newaxis]

    cos_a = np.cos(alpha)
    cos_b = np.cos(beta)

    sin_a = np.sin(alpha)
    sin_b = np.sin(beta)

    phi = np.arccos(cos_a * cos_b - mz * sin_a * sin_b)

    n0n1 = (mx ** 2 * (1 - cos_a) * (1 - cos_b) /
                (1 + cos_a * cos_b - mz * sin_a * sin_b))
    
    M = (1 - n0n1* np.sin(N * phi / 2) ** 2)
    coherence_signal_one_spin = M.flatten()
    
    return coherence_signal_one_spin



def calculate_coherence(spin_list, hf_df, exp_params):
    '''
    input:
        spin_list (1d numpy array of ints): indices of spins in hf_df
        hf_df (dataframe): hyperfine couplings and locations of all nuclear spins
        exp_params (list of dicts): each dict contains number of pulses (int), 
        timepoints (1d numpy array), magnetic field (G), and amount of noise (sigma^2)
    output:
        coherence_signals (list of 1d numpy array): each entry in the list corresponds to coherence
        signal
    '''
    
    # extract hyperfine parameters corresponding to spins in spin list
    A_par_list = []
    A_perp_list = []
    for spin in spin_list:
        A_par_list.append(hf_df.iloc[spin]['A_par'])
        A_perp_list.append(hf_df.iloc[spin]['A_perp'])
        
    # calculate coherence signals corresponding to each experiment    
    coherence_signals = []
    for index in range(exp_params['num_experiments']):
        coherence_signal = []
        timepoints = exp_params['timepoints'][index]
        mag_field = exp_params['mag_field'][index]
        num_pulses = exp_params['num_pulses'][index]
        for A_par, A_perp in zip(A_par_list, A_perp_list):
            coherence_signal.append(coherence_one_spin(timepoints, A_par, A_perp,
                                                       num_pulses, mag_field))
        coherence_signals.append(np.prod(np.array(coherence_signal), axis=0))
    
    return coherence_signals


def generate_trial_data(num_spins, hf_df, exp_params):
    '''
    input:
        num_spins (int): number of spins to simulate
        hf_df (dataframe): dataframe with hyperfine couplings and locations of spins
        exp_params (list of dicts): each dict contains number of pulses (int), 
        timepoints (1d numpy array), magnetic field (G), and amount of noise (sigma^2)
    output:
        spin_list (1d numpy array of ints): spins used to generate data
        coherence_data (list of 1d numpy array): each entry corresponds to an experiment
        in exp_pararms
    '''
    
    # randomly sample spins from hf_df
    spin_indices = np.arange(len(hf_df))
    spin_list = np.random.choice(spin_indices, size=num_spins, replace=False)
    
    # get coherence signals without noise
    coherence_signals_no_noise = calculate_coherence(spin_list, hf_df, exp_params)
    
    # add in noise
    coherence_signals = []
    for index in range(exp_params['num_experiments']):
        coherence_signal = (coherence_signals_no_noise[index] +
                            np.random.normal(scale=exp_params['noise'][index], 
                                             size=len(coherence_signals_no_noise[index])))
        coherence_signals.append(coherence_signal)
                          
    return spin_list, coherence_signals



def get_spins_in_neighborhood(spin, r, hf_dist_mat):
    '''
    input: 
        curr_spin (integer): index corresponding to spin in hf_dist_mat and hf_df
        r (float): distance in real space (A)
        hf_dist_mat (2d numpy array of floats): distance between spins (i) and (j)
    output:
        indices (1d numpy array of integers): list of integers of spin indices
    '''
    mask = hf_dist_mat[spin] < r
    indices = np.where(mask)[0]
    return indices


def get_error_spin_data(coherence_data, spin_list, exp_params, hf_df):
    '''
    input:
        data (list of 1d numpy array of floats): coherence data for selecting over
        spin_list (1d numpy array of ints): list of spin indices corresponding to hf_df
        exp_params (dict containing lists): contain pulse number, time points, and magnetic field
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
    output:
        error (float): L2 error between generated coherence data from spin list and
        exp params and the input coherence data
    '''
    
    simulated_coherence_signals = calculate_coherence(spin_list, hf_df, exp_params)
    
    err = 0
    for index in range(exp_params['num_experiments']):
        err += np.linalg.norm(coherence_data[index] - simulated_coherence_signals[index])
    
    return err


def sort_spins_indices(spin_list):
    '''
    input:
        spin_list (1d numpy array of ints): spin indices corresponding to hf_df
    output:
        sorted_spin_list (1d numpy array of ints): spin indices conrresponding to hf_df
    '''
    sorted_spin_list = spin_list.copy()
    sorted_spin_list.sort()
    return sorted_spin_list


def sort_spins_hf_mag(spin_list, hf_df):
    '''
    input:
        spin_list (1d numpy array of ints): spin indices corresponding to hf_df
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
    output:
        sorted_spin_list (1d numpy array of ints): spin indices conrresponding to hf_df
    '''
    return None


def jump_bool_uniform(current_k, k_max):
    '''
    output:
        jump (boolean): if TRUE, jump dimensions, otherwise do a within-model step
    '''
    if current_k == 1 and k_max == 1: # there are no valid jumps in this scenario
        return False
    
    if np.random.uniform(0, 1) < 0.5:
        return True
    else:
        return False


def birth_bool_uniform(current_k, k_max):
    '''
    input:
        current_k (int): current number of spins
    output:
        birth (boolean): if TRUE, add a dimension, otherwise remove a dimension
        
    '''
#     if current_k == 1 and k_max == 1:
#         print('should not be changing dimension')
#         return False
    
    if current_k == k_max: # cannot add a dimension beyond k_max
        return False
    elif current_k == 1: # cannot remove a dimension beyond 0
        return True
    else: # if in the middle then 50-50 add or remove a dimension
        if np.random.uniform(0, 1) < 0.5:
            return True
        else:
            return False


def within_model_step_RWMH(current_spins, r, hf_df, hf_dist_mat, 
                           coherence_data, exp_params, sigma_sq=None):
    '''
    input:
        current_spins (1d numpy array of ints of length current_k): indices of current spins
        r (float): radius for random walk
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
        hf_dist_mat (2d numpy array of floats): distance (A) between spins i and j
        coherence_data (1d numpy array of floats): coherence data 
        exp_params (list of dicts): containing number of pulses, time points taken, amount of noise,
        and magnetic field for each experiment resulting in coherence data
    output:
        next_spins (1d numpy array of ints of length current_k): indices of spins for next step
        in RWMH algorithm
        error (float): L2 error between coherence data and next_spins
    '''
    
    proposed_spins = get_proposal_spins_within_model_step_RWMH(current_spins, hf_dist_mat, r)
    
    log_a = get_log_accept_prob_within_model_step_RWMH(current_spins, proposed_spins, exp_params,
                                                 coherence_data, r, hf_df, hf_dist_mat, sigma_sq)
    u = np.random.uniform(0, 1)
    if np.log(u) < log_a:
        next_spins = proposed_spins
        error = get_error_spin_data(coherence_data, next_spins, exp_params, hf_df)
    else:
        next_spins = current_spins
        error = get_error_spin_data(coherence_data, next_spins, exp_params, hf_df)
    
    return next_spins, error


def remove_elements(original_list, elements_to_remove):
    return [element for element in original_list if element not in elements_to_remove]


def get_proposal_spins_within_model_step_RWMH(current_spins, hf_dist_mat, r):
    '''
    input:
        current_spins (1d numpy array of ints of length current_k): indices of current spins
        hf_dist_mat (2d numpy array of floats): distance (A) between spins i and j
        r (float): radius of neighbors we are considering
    output:
        proposal_spins(1d numpy array of ints of length current_k): indices of proposed spins
    '''
    
    # get neighbors for each of the current spins
    neighbors_all_spins = []
    for spin in current_spins:
        neighbors_all_spins.append(get_spins_in_neighborhood(spin, r, hf_dist_mat))
        
    prop_spins = []
    for neighbors in neighbors_all_spins:
        # make sure two walkers cannot be at the same spin at the same time
        filtered_neighbors = remove_elements(neighbors, prop_spins) 
        prop_spin = random.choice(filtered_neighbors)
        prop_spins.append(prop_spin)
    
    return prop_spins


def get_proposal_spins_birth_step(current_spins, hf_dist_mat):
    '''
    input:
        current_spins (1d numpy array of ints of length current_k): indices of current spins
        hf_dist_mat (2d numpy array of floats): distance (A) between spins i and j
    output:
        proposal_spins(1d numpy array of ints of length current_k): indices of proposed spins, 
        which are the same as the current spins plus an additional spin
    '''
    all_spins = list(range(len(hf_dist_mat)))
    filtered_spins = remove_elements(all_spins, current_spins)
    prop_spins = list(current_spins)
    prop_spins.append(random.choice(filtered_spins))
    return prop_spins


def get_proposal_spins_death_step(current_spins):
    '''
    input:
        current_spins (1d numpy array of ints of length current_k): indices of current spins
    output:
        proposal_spins(1d numpy array of ints of length current_k): indices of proposed spins, 
        which are the same as the current spins with one of the indices removed
    '''
    prop_spins = list(current_spins)
    index_to_remove = random.randint(0, len(current_spins)-1)
    prop_spins.pop(index_to_remove)
    return prop_spins


def get_log_prob_within_step_move(start_spins, end_spins, r, hf_dist_mat):
    '''
    input:
        start_spins (1d numpy array of ints): list of indices
        end_spins (1d numpy array of ints same length as start_spins): list of indices
        r (float): radius (A) for random walk
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
        hf_dist_mat (2d numpy array of floats): distance (A) between spins i and j
    output: 
        prob (float): log of probability of proposing a move between start_spins
        and end_spins
    '''
    
    abbrev_spins = []
    num_eligible_neighbors = []
    for start_spin, end_spin in zip(start_spins, end_spins):
        neighbors = get_spins_in_neighborhood(start_spin, r, hf_dist_mat)
        filtered_neighbors = remove_elements(neighbors, abbrev_spins)
        abbrev_spins.append(end_spin)
        num_eligible_neighbors.append(len(filtered_neighbors))
    
    # probability is 1/num neighbors multiplied together
    log_prob = 0
    for number in num_eligible_neighbors:
        log_prob -= np.log(number)  
    
    return log_prob


def get_log_likelihood_of_spins_given_data(spin_list, coherence_data, hf_df, exp_params, sigma_sq=None):
    '''
    input:
        spin_list (1d numpy array of ints): list of spin indices
        coherence_data (list of 1d numpy array of floats): coherence data
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
        exp_params (list of dicts): containing number of pulses, time points, magnitude of noise,
        and magnetic field (G)
        sigma_sq (float): amount of noise to use in likelihood calculation
    output:
        log_likelihood (float): log likelihood of the spins given data based on gaussian model 
        of noise
    '''
    
    simulated_coherence_signals = calculate_coherence(spin_list, hf_df, exp_params)
    
    log_likelihood = 0
    for index in range(exp_params['num_experiments']):
        # if default is None, use experimental value
        if sigma_sq == None:
            sigma_sq = exp_params['noise'][index]
   
        log_likelihood += np.sum(-1*np.square(coherence_data[index]-
                                              simulated_coherence_signals[index]))/(2*sigma_sq)
    
    return log_likelihood


def birth_step(current_k, current_spins, r, hf_df, hf_dist_mat, 
                           coherence_data, exp_params, k_max):
    '''
    input:
        current_k (int): number of spins in current model
        current_spins (1d numpy array of ints of length current_k): indices of current spins
        r (float): radius for random walk
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
        hf_dist_mat (2d numpy array of floats): distance (A) between spins i and j
        coherence_data (1d numpy array of floats): coherence data 
        exp_params (list of dicts): containing number of pulses, time points taken, amount of noise,
        and magnetic field for each experiment resulting in coherence data
        k_max (int): maximum number of spins considering
    output:
        next_k (int): number of next_spins
        next_error (float): L2 error between coherence data and next_spins
        next_spins (1d numpy array of ints of length current_k): indices of spins for next step
        in RWMH algorithm
    '''
    
    proposed_spins = get_proposal_spins_birth_step(current_spins, hf_dist_mat)
    log_a = get_log_accept_prob_jump_step(current_spins, proposed_spins, exp_params,
                                          coherence_data, hf_df, k_max)
    
    u = np.random.uniform(0, 1)
    if np.log(u) < log_a:
        next_spins = proposed_spins
    else:
        next_spins = current_spins
    
    error = get_error_spin_data(coherence_data, next_spins, exp_params, hf_df)
    next_k = len(next_spins)
    
    return next_k, next_spins, error



def death_step(current_k, current_spins, r, hf_df, hf_dist_mat, 
                           coherence_data, exp_params, k_max):
    '''
    input:
        current_k (int): number of spins in current model
        current_spins (1d numpy array of ints of length current_k): indices of current spins
        r (float): radius for random walk
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
        hf_dist_mat (2d numpy array of floats): distance (A) between spins i and j
        coherence_data (1d numpy array of floats): coherence data 
        exp_params (list of dicts): containing number of pulses, time points taken, amount of noise,
        and magnetic field for each experiment resulting in coherence data
    output:
        next_k (int): number of next_spins
        next_error (float): L2 error between coherence data and next_spins
        next_spins (1d numpy array of ints of length current_k): indices of spins for next step
        in RWMH algorithm
    '''
    proposed_spins = get_proposal_spins_death_step(current_spins)
    log_a = get_log_accept_prob_jump_step(current_spins, proposed_spins, exp_params,
                                          coherence_data, hf_df, k_max)
    
    u = np.random.uniform(0, 1)
    if np.log(u) < log_a:
        next_spins = proposed_spins
    else:
        next_spins = current_spins
        
    error = get_error_spin_data(coherence_data, next_spins, exp_params, hf_df)
    next_k = len(next_spins)
    
    return next_k, next_spins, error


def get_log_accept_prob_within_model_step_RWMH(current_spins, proposed_spins, exp_params,
                                                 coherence_data, r, hf_df, hf_dist_mat, sigma_sq=None):
    '''
    input:
        current_spins (1d numpy array of ints):
        proposed_spins (1d numpy array of ints):
        exp_params (list of dicts): containing number of pulses, time points taken, amount of noise,
        and magnetic field for each experiment resulting in coherence data
        coherence_data (1d numpy array of floats): coherence data
        r (float): radius for random walk
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
        hf_dist_mat (2d numpy array of floats): distance (A) between spins i and j
        sigma_sq (float): amount of noise to assume for likelihood (default is use same value as experimental data)
    output:
        log_a(float): log of acceptance probability to accept within-model move
        between current spins and proposed_spins
    '''
    
    log_L_prop_spins = get_log_likelihood_of_spins_given_data(proposed_spins, coherence_data, 
                                                              hf_df, exp_params, sigma_sq=None)
    log_L_curr_spins = get_log_likelihood_of_spins_given_data(current_spins, coherence_data, 
                                                              hf_df, exp_params, sigma_sq=None)
    log_prob_prop_to_curr = get_log_prob_within_step_move(proposed_spins, current_spins, r, 
                                                          hf_dist_mat)
    log_prob_curr_to_prop = get_log_prob_within_step_move(current_spins,proposed_spins, r, 
                                                          hf_dist_mat)
    
    log_a = min((0, (log_L_prop_spins + log_prob_prop_to_curr -
                     log_L_curr_spins + log_prob_curr_to_prop)))
    
    return log_a


def get_birth_prob(k, k_max):
    '''
    input:
        k (int): dimension
        k_max (int): maximum number of spins allowed
    output:
        b_k (float): birth probability
    '''
    if k == 1:
        b_k = 1
    elif k == k_max:
        b_k = 1e-20
    else:
        b_k = 0.5
        
    return b_k
    
    
def get_death_prob(k, k_max):
    '''
    input:
        k (int): dimension
        k_max (int): maximum number of spins allowed
    output:
        d_k (float): death probability
    '''
    if k == 1:
        d_k = 1e-20
    elif k == k_max:
        d_k = 1
    else:
        d_k = 0.5      
    return d_k
    


def get_log_accept_prob_jump_step(current_spins, proposed_spins, exp_params,
                                  coherence_data, hf_df, k_max, sigma_sq=None):
    '''
    input:
        current_spins (1d numpy array of ints):
        proposed_spins (1d numpy array of ints):
        exp_params (list of dicts): containing number of pulses, time points taken, amount of noise,
        and magnetic field for each experiment resulting in coherence data
        coherence_data (list of 1d numpy array of floats): coherence data
        k_max (int): maximum number of spins allowed in system
    output:
       log_a (float): log of acceptance probability to accept jump move
    '''
    
    log_L_prop_spins = get_log_likelihood_of_spins_given_data(proposed_spins, coherence_data,
                                                              hf_df, exp_params, sigma_sq)
    log_L_curr_spins = get_log_likelihood_of_spins_given_data(current_spins, coherence_data,
                                                              hf_df, exp_params, sigma_sq)
    
    curr_k = len(current_spins)
    prop_k = len(proposed_spins)
    
    if prop_k > curr_k: # birth step
        log_d_k = 0 # np.log(get_death_prob(prop_k, k_max))
        log_b_k = 0 # np.log(get_birth_prob(curr_k, k_max))
        log_a = min((0, log_L_prop_spins + log_d_k - log_L_curr_spins - log_b_k))
    else: # death step
        log_b_k = 0 # np.log(get_birth_prob(prop_k, k_max))
        log_d_k = 0 # np.log(get_death_prob(curr_k, k_max))
        log_a = min((0, log_L_prop_spins + log_b_k - log_L_curr_spins - log_d_k))
        
    return log_a


def RWMH_no_jumps(initial_spin_list, hf_df, hf_dist_mat, r, exp_params, 
                  coherence_data, num_trials):
    '''
    input:
        initial_spin_list (1d numpy array of ints):
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
        hf_dist_mat (2d numpy array of floats): distance (A) between spins i and j
        r (float): radius for random walk
        exp_params (list of dicts): containing number of pulses, time points taken, amount of noise,
        and magnetic field for each experiment resulting in coherence data
        coherence_data (list of 1d numpy array of floats): coherence data
        num_trials (int): number of steps in random walk
    output:
        spin_samples (list of 1d numpy array of ints): list of random walk (length num_trials)
        error_samples (list of floats): error associated with each step 
    '''
    
    spin_samples = []
    error_samples = []
    spin_samples.append(initial_spin_list)
    error_initial_spins = get_error_spin_data(coherence_data, initial_spin_list, exp_params, hf_df)
    error_samples.append(error_initial_spins)
    count = 1
    
    while count < num_trials:
        current_spins = spin_samples[count-1]
        next_spins, error = within_model_step_RWMH(current_spins, r, hf_df, hf_dist_mat, 
                                                   coherence_data, exp_params)
        spin_samples.append(next_spins)
        error_samples.append(error)
        count += 1
        
    return spin_samples, error_samples


def RJMCMC_RWMH(initial_spin_list, hf_df, hf_dist_mat, r, exp_params,
                coherence_data, num_trials, k_max, sigma_sq=None):
    '''
    input:
        initial_spin_list (1d numpy array of ints):
        hf_df (dataframe): dataframe containing hyperfine couplings and location of 
        all nuclear spins
        hf_dist_mat (2d numpy array of floats): distance (A) between spins i and j
        r (float): radius for random walk
        exp_params (list of dicts): containing number of pulses, time points taken, amount of noise,
        and magnetic field for each experiment resulting in coherence data
        coherence_data (list of 1d numpy array of floats): coherence data
        num_trials (int): number of steps in random walk
        sigma_sq (float): amount of noise to assume for likelihood (default is use same value as experimental data)
    output:
        k_samples (list of ints): list of number of spins associated with each step
        spin_samples (list of 1d numpy array of ints): list of random walk (length num_trials)
        error_samples (list of floats): error associated with each step 
    '''
    
    spin_samples = []
    error_samples = []
    k_samples = []
    
    error_initial_spins = get_error_spin_data(coherence_data, initial_spin_list, exp_params, hf_df)
    spin_samples.append(initial_spin_list)
    error_samples.append(error_initial_spins)
    k_samples.append(len(initial_spin_list))
    count = 1
    
    while count < num_trials:
        
        current_spins = spin_samples[count-1]
        current_k = k_samples[count-1]
        
        if jump_bool_uniform(current_k, k_max): # jump dimensions
            
            # print(f'current_k: {current_k}, k_max:{k_max}, True', flush=True)
            
            if birth_bool_uniform(current_k, k_max): # birth step 
                next_k, next_spins, error = birth_step(current_k, current_spins, r, hf_df, hf_dist_mat, 
                                                       coherence_data, exp_params, k_max)
                # print('birth')
            else: # death step
                next_k, next_spins, error = death_step(current_k, current_spins, r, hf_df, hf_dist_mat, 
                                                       coherence_data, exp_params, k_max)
                # print('death')
        else: # within model RWMH step, same k
            
            # print(f'current_k: {current_k}, k_max:{k_max}, False', flush=True)  
            next_spins, error = within_model_step_RWMH(current_spins, r, hf_df, hf_dist_mat, 
                                                       coherence_data, exp_params, sigma_sq)
        
        spin_samples.append(next_spins)
        error_samples.append(error)
        k_samples.append(len(next_spins))
        count += 1
    
    return k_samples, spin_samples, error_samples



# to run on Midway
def random_trial_two_pulses(HF_FILE, BOOTSTRAPPED_DF_FILE, CONF, HF_THRESH_HIGH, NOISE_8, 
                            NOISE_16, NUM_SPINS, K_MAX, NUM_TRIALS, R, NUM_ENSEMBLES):
    
    '''
    input:
        HF_FILE (string): file containing hf parameters and locations of atoms
        BOOTSTRAPPED_DF_FILE (string): file containing bootstrapped data
        CONF (float): confidence, needs to be in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
        HF_THRESH_HIGH (float): upper limit cutoff of spins
        NOISE_8 (float): norm of uncertainty vector for 8-pulse experiment
        NOISE_16 (float): norm of uncertainty vector for 8-pulse experiment
        NUM_SPINS (int): number of spins to simulate
        K_MAX (int): upper limit on number of spins to simulate
        NUM_TRIALS (int): how many steps the walkers should take
        R (float): hyperparameter of distance of step each walker could take
        NUM_ENSEMBLES (int): number of different initializations to start form
    output:
        ensembles (list of dicts): contains dict for each ensemble about walkers
        spin_list_ground (list of ints): ground truth spins used to generate data
        exp_params (dict): containing experimental parameters
    '''
    noise = NOISE_8 + NOISE_16
    bootstrapped_df = pd.read_pickle(BOOTSTRAPPED_DF_FILE)
    hf_thresh_low = get_hf_limit(bootstrapped_df, 'joint', noise, CONF)
    print(str(hf_thresh_low), flush=True)
    
    hf_df = make_df_from_Ivady_file(HF_FILE, HF_THRESH_HIGH, hf_thresh_low)
    print('making dist matrix', flush=True)
    hf_dist_mat = get_distance_matrix(hf_df)
    print('finished making dist matrix', flush=True)
    print('num spins: '+str(len(hf_dist_mat)), flush=True)
    
    _, _, _, TIME_8 = get_specific_exp_parameters(8)
    _, _, _, TIME_16 = get_specific_exp_parameters(16)
    
    # come up with some sample data
    num_experiments = 2
    num_pulses = [8, 16]
    mag_field = [311, 311]
    
    noise = [np.sqrt(np.linalg.norm(NOISE_8**2/len(TIME_8))),
             np.sqrt(np.linalg.norm(NOISE_16**2/len(TIME_16)))]
    time = [TIME_8, TIME_16]
    
    exp_params = make_exp_params_dict(num_experiments, num_pulses, mag_field, noise, time)
    spin_list_ground, coherence_signals = generate_trial_data(NUM_SPINS, hf_df, exp_params)
    
    ensembles = []
    num_ensembles = 0
    while num_ensembles < NUM_ENSEMBLES:
        print(num_ensembles)
        ensemble_dict = {}
        num_spins_initial = np.random.choice(range(1, K_MAX+1))
        spin_indices = np.arange(len(hf_df))
        spin_list_initial = np.random.choice(spin_indices, size=num_spins_initial, replace=False)
    
        k_trials, spin_trials, error_trials = RJMCMC_RWMH(spin_list_initial, hf_df, hf_dist_mat, R,
                                                      exp_params, coherence_signals, NUM_TRIALS, 
                                                      K_MAX)
        ensemble_dict['initial_spins'] = spin_list_initial
        ensemble_dict['k_trials'] = k_trials
        ensemble_dict['spin_trials'] = spin_trials
        ensemble_dict['error_trials'] = error_trials
        ensembles.append(ensemble_dict)
        num_ensembles += 1
    
    return ensembles, spin_list_ground, exp_params


# to run on Midway
def random_trial_two_pulses_look_at_noise(HF_FILE, BOOTSTRAPPED_DF_FILE, CONF, HF_THRESH_HIGH, HF_THRESH_LOW, NOISE_8, 
                            NOISE_16, L_NOISE, NUM_SPINS, K_MAX, NUM_TRIALS, R, NUM_ENSEMBLES):
    
    '''
    input:
        HF_FILE (string): file containing hf parameters and locations of atoms
        BOOTSTRAPPED_DF_FILE (string): file containing bootstrapped data
        CONF (float): confidence, needs to be in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
        HF_THRESH_HIGH (float): upper limit cutoff of spins
        HF_LOW (float): lower limit cutoff of spins
        NOISE_8 (float): amount of noise to add to simulations at each time point (eg add np.random.noise(0, NOISE_8)
        NOISE_16 (float): amount of noise to add to simulations at each time point (eg add np.random.noise(0, NOISE_8)
        L_NOISE (float): amount of noise to use in likelihood calculation
        NUM_SPINS (int): number of spins to simulate
        K_MAX (int): upper limit on number of spins to simulate
        NUM_TRIALS (int): how many steps the walkers should take
        R (float): hyperparameter of distance of step each walker could take
        NUM_ENSEMBLES (int): number of different initializations to start form
    output:
        ensembles (list of dicts): contains dict for each ensemble about walkers
        spin_list_ground (list of ints): ground truth spins used to generate data
        exp_params (dict): containing experimental parameters
    '''
    
    hf_df = make_df_from_Ivady_file(HF_FILE, HF_THRESH_HIGH, HF_THRESH_LOW)
    print('making dist matrix', flush=True)
    hf_dist_mat = get_distance_matrix(hf_df)
    print('finished making dist matrix', flush=True)
    print('num spins: '+str(len(hf_dist_mat)), flush=True)
    
    _, _, _, TIME_8 = get_specific_exp_parameters(8)
    _, _, _, TIME_16 = get_specific_exp_parameters(16)
    
    # come up with some sample data
    num_experiments = 2
    num_pulses = [8, 16]
    mag_field = [311, 311]
    
    noise = [NOISE_8, NOISE_16]
    time = [TIME_8, TIME_16]
    
    sigma_sq = L_NOISE
    
    exp_params = make_exp_params_dict(num_experiments, num_pulses, mag_field, noise, time)
    spin_list_ground, coherence_signals = generate_trial_data(NUM_SPINS, hf_df, exp_params)
    
    ensembles = []
    num_ensembles = 0
    while num_ensembles < NUM_ENSEMBLES:
        print(num_ensembles)
        ensemble_dict = {}
        num_spins_initial = np.random.choice(range(1, K_MAX+1))
        spin_indices = np.arange(len(hf_df))
        spin_list_initial = np.random.choice(spin_indices, size=num_spins_initial, replace=False)
    
        k_trials, spin_trials, error_trials = RJMCMC_RWMH(spin_list_initial, hf_df, hf_dist_mat, R,
                                                      exp_params, coherence_signals, NUM_TRIALS, 
                                                      K_MAX, sigma_sq)
        ensemble_dict['initial_spins'] = spin_list_initial
        ensemble_dict['k_trials'] = k_trials
        ensemble_dict['spin_trials'] = spin_trials
        ensemble_dict['error_trials'] = error_trials
        ensembles.append(ensemble_dict)
        num_ensembles += 1
    
    return ensembles, spin_list_ground, exp_params




########### below here need to be documented and re-written
def get_dict_data(data, num_pulses, nv_num):
    
    data_dict = {}
    
    data_dict['N'] = 1
    data_dict['nv_num'] = nv_num
    data_dict['mag_field'] = 311 # G
    data_dict['num_lines'] = 504 # number of time points
    data_dict['read_time'] = 600 # ns
    data_dict['pi_time'] = 120 # ns
    
    if num_pulses == 8:
        data_dict['XYn'] = 8 # number of pulses
        data_dict['t0'] = 6 # us
        data_dict['dt'] = 0.48 # us
        data_dict['t_tot'] = 126 # us
        data_dict['pi_time'] = 120 # ns
        
    elif num_pulses == 16:
        data_dict['XYn'] = 16 # number of pulses
        data_dict['t0'] = 12 # us
        data_dict['dt'] = 0.96 # us
        data_dict['t_tot'] = 252 # us

    time, exp_data = get_plot_data(data)
    
    data_dict['time'] = time
    data_dict['exp_data'] = exp_data
    data_dict['rescaled_data'] = 2*(exp_data-0.5)
    
    return data_dict


def get_plot_data(data):
    
    # DIRECTLY FROM CHRIS
    pz_channel = 6 # ch6 TFT -> matlab data col 7 
    mz_channel = 5 # ch5 TTT -> matlab data col 6

    pz_channel_norm = 3 # TFF #matlab indexing so -1 more???
    mz_channel_norm = 1 # TTF #matlab indexing so -1 more???

    n = int((data.shape[0]-4)/2)
    data.rename(columns = {0:'time'}, inplace=True)
    t=data['time'].to_numpy()[:n]

    pz_data = (data[pz_channel].to_numpy()[:n]+data[mz_channel].to_numpy()[n:2*n])/1000
    mz_data = (data[mz_channel].to_numpy()[n-1::-1]+data[pz_channel].to_numpy()[2*n-1:n-1:-1])/1000

    # normalization, e.g. line 52, bright channel
    # Use short time read-outs
    high_norm = (data[pz_channel].to_numpy()[len(data)-4]+data[mz_channel].to_numpy()[len(data)-3])/1000; 

    # normalization, e.g. line 51, dark channel
    # Use short time read-out
    low_norm = (data[mz_channel].to_numpy()[len(data)-2]+data[pz_channel].to_numpy()[len(data)-1])/1000; 
    
    return t,  (pz_data-mz_data)/(2*abs(high_norm-low_norm)) + 0.5 #P_z projection?
    # return t,  (pz_data-mz_data)/(abs(high_norm-low_norm)) # coherence signal? [-1, 1]
    


def fit_exp_data_rjmcmc(HF_FILE, BOOTSTRAPPED_DF_FILE, CONF, HF_THRESH_HIGH, NOISE_8,
                        NOISE_16, K_MAX, NUM_TRIALS, R, NUM_ENSEMBLES, DATA_PATH_8, DATA_PATH_16,
                        NV_NUM):
    
    # info from experimental data
    data_csv_8 = pd.read_csv(DATA_PATH_8, sep='\t', header=None)
    data_csv_16 = pd.read_csv(DATA_PATH_16, sep='\t', header=None)
    data_dict_8 = get_dict_data(data_csv_8, 8, NV_NUM)
    data_dict_16 = get_dict_data(data_csv_16, 16, NV_NUM)
    data_8 = data_dict_8['rescaled_data']
    data_16 = data_dict_16['rescaled_data']
    
    _, _, _, TIME_8 = get_specific_exp_parameters(8)
    _, _, _, TIME_16 = get_specific_exp_parameters(16)
    
    num_experiments = 2
    num_pulses = [8, 16]
    mag_field = [311, 311]
    noise = [np.sqrt(np.linalg.norm(NOISE_8**2/len(TIME_8))),
             np.sqrt(np.linalg.norm(NOISE_16**2/len(TIME_16)))]
    time = [TIME_8, TIME_16]
    
    exp_params = make_exp_params_dict(num_experiments, num_pulses, mag_field, noise, time)
    coherence_signals = []
    coherence_signals.append(data_8)
    coherence_signals.append(data_16)
    
    # ab initio hf data and bootstrap df
    noise = NOISE_8 + NOISE_16
    bootstrapped_df = pd.read_pickle(BOOTSTRAPPED_DF_FILE)
    hf_thresh_low = get_hf_limit(bootstrapped_df, 'joint', noise, CONF)
    
    hf_df = make_df_from_Ivady_file(HF_FILE, HF_THRESH_HIGH, hf_thresh_low)
    hf_dist_mat = get_distance_matrix(hf_df)
    
    ensembles = []
    num_ensembles = 0
    while num_ensembles < NUM_ENSEMBLES:
        print(num_ensembles)
        ensemble_dict = {}
        num_spins_initial = np.random.choice(range(1, K_MAX+1))
        spin_indices = np.arange(len(hf_df))
        spin_list_initial = np.random.choice(spin_indices, size=num_spins_initial, replace=False)
    
        k_trials, spin_trials, error_trials = RJMCMC_RWMH(spin_list_initial, hf_df, hf_dist_mat, R,
                                                      exp_params, coherence_signals, NUM_TRIALS, 
                                                      K_MAX)
        ensemble_dict['initial_spins'] = spin_list_initial
        ensemble_dict['k_trials'] = k_trials
        ensemble_dict['spin_trials'] = spin_trials
        ensemble_dict['error_trials'] = error_trials
        ensembles.append(ensemble_dict)
        num_ensembles += 1
    
    return ensembles, exp_params, coherence_signals, hf_df
                                 
                                 
                                 