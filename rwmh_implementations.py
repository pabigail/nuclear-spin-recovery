import rjmcmc
import pandas as pd
import numpy as np

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
    error_initial_spins = rjmcmc.get_error_spin_data(coherence_data, initial_spin_list, exp_params, hf_df)
    error_samples.append(error_initial_spins)
    count = 1
    
    while count < num_trials:
        current_spins = spin_samples[count-1]
        next_spins, error = rjmcmc.within_model_step_RWMH(current_spins, r, hf_df, hf_dist_mat, 
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
    
    error_initial_spins = rjmcmc.get_error_spin_data(coherence_data, initial_spin_list, exp_params, hf_df)
    spin_samples.append(initial_spin_list)
    error_samples.append(error_initial_spins)
    k_samples.append(len(initial_spin_list))
    count = 1
    
    while count < num_trials:
        
        current_spins = spin_samples[count-1]
        current_k = k_samples[count-1]
        
        if rjmcmc.jump_bool_uniform(current_k, k_max): # jump dimensions
            
            # print(f'current_k: {current_k}, k_max:{k_max}, True', flush=True)
            
            if rjmcmc.birth_bool_uniform(current_k, k_max): # birth step 
                next_k, next_spins, error = rjmcmc.birth_step(current_k, current_spins, r, hf_df, hf_dist_mat, 
                                                       coherence_data, exp_params, k_max)
                # print('birth')
            else: # death step
                next_k, next_spins, error = rjmcmc.death_step(current_k, current_spins, r, hf_df, hf_dist_mat, 
                                                       coherence_data, exp_params, k_max)
                # print('death')
        else: # within model RWMH step, same k
            
            # print(f'current_k: {current_k}, k_max:{k_max}, False', flush=True)  
            next_spins, error = rjmcmc.within_model_step_RWMH(current_spins, r, hf_df, hf_dist_mat, 
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
    hf_thresh_low = rjmcmc.get_hf_limit(bootstrapped_df, 'joint', noise, CONF)
    print(str(hf_thresh_low), flush=True)
    
    hf_df = rjmcmc.make_df_from_Ivady_file(HF_FILE, HF_THRESH_HIGH, hf_thresh_low)
    print('making dist matrix', flush=True)
    hf_dist_mat = rjmcmc.get_distance_matrix(hf_df)
    print('finished making dist matrix', flush=True)
    print('num spins: '+str(len(hf_dist_mat)), flush=True)
    
    _, _, _, TIME_8 = rjmcmc.get_specific_exp_parameters(8)
    _, _, _, TIME_16 = rjmcmc.get_specific_exp_parameters(16)
    
    # come up with some sample data
    num_experiments = 2
    num_pulses = [8, 16]
    mag_field = [311, 311]
    
    noise = [np.sqrt(np.linalg.norm(NOISE_8**2/len(TIME_8))),
             np.sqrt(np.linalg.norm(NOISE_16**2/len(TIME_16)))]
    time = [TIME_8, TIME_16]
    
    exp_params = rjmcmc.make_exp_params_dict(num_experiments, num_pulses, mag_field, noise, time)
    spin_list_ground, coherence_signals = rjmcmc.generate_trial_data(NUM_SPINS, hf_df, exp_params)
    
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
    
    hf_df = rjmcmc.make_df_from_Ivady_file(HF_FILE, HF_THRESH_HIGH, HF_THRESH_LOW)
    print('making dist matrix', flush=True)
    hf_dist_mat = rjmcmc.get_distance_matrix(hf_df)
    print('finished making dist matrix', flush=True)
    print('num spins: '+str(len(hf_dist_mat)), flush=True)
    
    _, _, _, TIME_8 = rjmcmc.get_specific_exp_parameters(8)
    _, _, _, TIME_16 = rjmcmc.get_specific_exp_parameters(16)
    
    # come up with some sample data
    num_experiments = 2
    num_pulses = [8, 16]
    mag_field = [311, 311]
    
    noise = [NOISE_8, NOISE_16]
    time = [TIME_8, TIME_16]
    
    sigma_sq = L_NOISE
    
    exp_params = rjmcmc.make_exp_params_dict(num_experiments, num_pulses, mag_field, noise, time)
    spin_list_ground, coherence_signals = rjmcmc.generate_trial_data(NUM_SPINS, hf_df, exp_params)
    
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


def fit_exp_data_rjmcmc(HF_FILE, BOOTSTRAPPED_DF_FILE, CONF, HF_THRESH_HIGH, NOISE_8,
                        NOISE_16, K_MAX, NUM_TRIALS, R, NUM_ENSEMBLES, DATA_PATH_8, DATA_PATH_16,
                        NV_NUM):
    
    # info from experimental data
    data_csv_8 = pd.read_csv(DATA_PATH_8, sep='\t', header=None)
    data_csv_16 = pd.read_csv(DATA_PATH_16, sep='\t', header=None)
    data_dict_8 = rjmcmc.get_dict_data(data_csv_8, 8, NV_NUM)
    data_dict_16 = rjmcmc.get_dict_data(data_csv_16, 16, NV_NUM)
    data_8 = data_dict_8['rescaled_data']
    data_16 = data_dict_16['rescaled_data']
    
    _, _, _, TIME_8 = rjmcmc.get_specific_exp_parameters(8)
    _, _, _, TIME_16 = rjmcmc.get_specific_exp_parameters(16)
    
    num_experiments = 2
    num_pulses = [8, 16]
    mag_field = [311, 311]
    noise = [np.sqrt(np.linalg.norm(NOISE_8**2/len(TIME_8))),
             np.sqrt(np.linalg.norm(NOISE_16**2/len(TIME_16)))]
    time = [TIME_8, TIME_16]
    
    exp_params = rjmcmc.make_exp_params_dict(num_experiments, num_pulses, mag_field, noise, time)
    coherence_signals = []
    coherence_signals.append(data_8)
    coherence_signals.append(data_16)
    
    # ab initio hf data and bootstrap df
    noise = NOISE_8 + NOISE_16
    bootstrapped_df = pd.read_pickle(BOOTSTRAPPED_DF_FILE)
    hf_thresh_low = rjmcmc.get_hf_limit(bootstrapped_df, 'joint', noise, CONF)
    
    hf_df = rjmcmc.make_df_from_Ivady_file(HF_FILE, HF_THRESH_HIGH, hf_thresh_low)
    hf_dist_mat = rjmcmc.get_distance_matrix(hf_df)
    
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


def RJMCMC_RWMH_with_parallel_tempering(initial_spin_list, hf_df, hf_dist_mat, r, exp_params,
                                        coherence_data, num_trials, k_max,
                                        num_strands, beta, num_rjmcmc_steps, num_parallel_steps,
                                        sigma_sq=None):
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
    
    error_initial_spins = rjmcmc.get_error_spin_data(coherence_data, initial_spin_list, exp_params, hf_df)
    spin_samples.append(initial_spin_list)
    error_samples.append(error_initial_spins)
    k_samples.append(len(initial_spin_list))
    count = 1
    
    while count < num_trials:
        
        count_rjmcmc = 0
        # rjmcmc steps
        while count_rjmcmc < num_rjmcmc_steps:
        
            current_spins = spin_samples[count-1]
            current_k = k_samples[count-1]
            # if jump_bool_uniform(current_k, k_max): # jump dimensions

            # print(f'current_k: {current_k}, k_max:{k_max}, True', flush=True)
            
            if rjmcmc.birth_bool_uniform(current_k, k_max): # birth step 
                next_k, next_spins, error = rjmcmc.birth_step(current_k, current_spins, r, hf_df, hf_dist_mat, 
                                                       coherence_data, exp_params, k_max)
                # print('birth')
            else: # death step
                next_k, next_spins, error = rjmcmc.death_step(current_k, current_spins, r, hf_df, hf_dist_mat, 
                                                       coherence_data, exp_params, k_max)
                # print('death')
        
            spin_samples.append(next_spins)
            error_samples.append(error)
            k_samples.append(len(next_spins))
            count += 1
            count_rjmcmc += 1
         
        # parallel tempering steps
        spin_samples_par, error_samples_par = rjmcmc.parallel_tempering_steps(spin_samples[count-1], hf_df, hf_dist_mat,
                                                                       exp_params,
                                                                   coherence_data, num_strands, num_parallel_steps,
                                                                   r, beta)
        for i in range(len(spin_samples_par)):
            spin_samples.append(spin_samples_par[0][i])
            error_samples.append(error_samples_par[0][i])
            k_samples.append(len(spin_samples_par[0][i]))
            count += 1
    
    return k_samples, spin_samples, error_samples