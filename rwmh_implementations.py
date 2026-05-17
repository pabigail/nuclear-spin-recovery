#!/usr/bin/env python

import rjmcmc
import pandas as pd
import numpy as np
import pickle as pkl

def RWMH_no_jumps_only_spins(initial_spin_list, hf_df, hf_dist_mat, r, exp_params, 
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
        spin_ind = (count % len(initial_spin_list))
        next_spins, error = rjmcmc.within_model_step_RWMH_one_spin(current_spins, r, hf_df, hf_dist_mat, 
                                                   coherence_data, exp_params, sigma_sq=None, beta_k=None, spin_ind=spin_ind)
        spin_samples.append(next_spins)
        error_samples.append(error)
        count += 1
        
    return spin_samples, error_samples


def RWMH_no_jumps_only_T2(initial_exp_params, hf_df, spin_list, r_T2,
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
    
    T2_samples = []
    error_samples = []
    
    T2_samples.append(initial_exp_params['T2'])
    error_initial_T2 = rjmcmc.get_error_spin_data(coherence_data, spin_list, initial_exp_params, hf_df)
    error_samples.append(error_initial_T2)
    exp_params = initial_exp_params
    
    count = 1
    while count < num_trials:
        current_T2 = T2_samples[count-1]
        
        exp_index = (count % initial_exp_params['num_experiments'])
        next_exp_params, error = rjmcmc.within_model_step_RWMH_T2(spin_list, r_T2, exp_index, hf_df,  
                                                   coherence_data, exp_params, sigma_sq=None, beta_k=None)
        exp_params = next_exp_params
        T2_samples.append(exp_params['T2'])
        error_samples.append(error)
        count += 1
        
    T2_samples = np.vstack(T2_samples)
        
    return T2_samples, error_samples


def RWMH_no_jumps_spins_and_T2(initial_exp_params, initial_spin_list, hf_df, r_spin, r_T2, hf_dist_mat,
                              coherence_data, num_trials, num_spin_steps, num_T2_steps):
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
    
    T2_samples = []
    spin_samples = []
    error_samples = []
    
    spin_samples.append(initial_spin_list)
    T2_samples.append(initial_exp_params['T2'])
    error_initial_T2 = rjmcmc.get_error_spin_data(coherence_data, initial_spin_list, initial_exp_params, hf_df)
    error_samples.append(error_initial_T2)
    exp_params = initial_exp_params
    
    count_total = 1
    count_spin_steps = 1
    count_T2_steps = 1
    
    while count_total < num_trials:
        
        # spin steps
        while count_spin_steps < num_spin_steps:
            
            current_spins = spin_samples[count_total-1]
            spin_ind = (count % len(initial_spin_list))
            next_spins, error = rjmcmc.within_model_step_RWMH_one_spin(current_spins, r_spin, hf_df, hf_dist_mat, 
                                                   coherence_data, exp_params, sigma_sq=None, beta_k=None, spin_ind=spin_ind)
            spin_samples.append(next_spins)
            T2_samples.append(exp_params['T2'])
            error_samples.append(error)
            count_total += 1
            count_spin_steps += 1
            
        # T2 steps
        while count_T2_steps < num_T2_steps:
        
            current_T2 = T2_samples[count_total-1]
            current_spins = spin_samples[count_total-1]
        
            exp_index = (count_total % initial_exp_params['num_experiments'])
            next_exp_params, error = rjmcmc.within_model_step_RWMH_T2(current_spins, r_T2, exp_index, hf_df,  
                                                   coherence_data, exp_params, sigma_sq=None, beta_k=None)
            exp_params = next_exp_params
            T2_samples.append(exp_params['T2'])
            spin_samples.append(current_spins)
            error_samples.append(error)
            count_total += 1
            count_T2_steps += 1
           
        
    T2_samples = np.vstack(T2_samples)
        
    return spin_samples, T2_samples, error_samples


def RJMCMC_RWMH_with_parallel_tempering(initial_spin_list, hf_df, hf_dist_mat, r_spin, r_T2, initial_exp_params,
                                        coherence_data, num_trials, k_max,
                                        num_strands, beta, num_rjmcmc_steps, num_parallel_steps, num_T2_steps,
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
    T2_samples = []
    
    error_initial_spins = rjmcmc.get_error_spin_data(coherence_data, initial_spin_list, initial_exp_params, hf_df)
    spin_samples.append(initial_spin_list)
    error_samples.append(error_initial_spins)
    k_samples.append(len(initial_spin_list))
    T2_samples.append(initial_exp_params['T2'])
    exp_params = initial_exp_params
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
                next_k, next_spins, error = rjmcmc.birth_step(current_k, current_spins, r_spin, hf_df, hf_dist_mat, 
                                                       coherence_data, exp_params, k_max, sigma_sq)
                # print('birth')
            else: # death step
                next_k, next_spins, error = rjmcmc.death_step(current_k, current_spins, r_spin, hf_df, hf_dist_mat, 
                                                       coherence_data, exp_params, k_max, sigma_sq)
                # print('death')
        
            spin_samples.append(next_spins)
            error_samples.append(error)
            k_samples.append(len(next_spins))
            T2_samples.append(exp_params['T2'])
            count += 1
            count_rjmcmc += 1
         
        
        # fit T2 steps
        count_T2 = 0
        while count_T2 < num_T2_steps:
            current_T2 = T2_samples[count-1]
            current_spins = spin_samples[count-1]
        
            exp_index = (count % exp_params['num_experiments'])
            next_exp_params, error = rjmcmc.within_model_step_RWMH_T2(current_spins, r_T2, exp_index, hf_df,  
                                                   coherence_data, exp_params, sigma_sq=sigma_sq, beta_k=None)
            exp_params = next_exp_params
            T2_samples.append(exp_params['T2'])
            spin_samples.append(current_spins)
            error_samples.append(error)
            k_samples.append(len(current_spins))
            count += 1
            count_T2 += 1
            
        
        # parallel tempering steps
        spin_samples_par, error_samples_par = rjmcmc.parallel_tempering_steps(spin_samples[count-1], hf_df, hf_dist_mat,
                                                                       exp_params,
                                                                   coherence_data, num_strands, num_parallel_steps,
                                                                   r_spin, beta, sigma_sq)
        for i in range(len(spin_samples_par)):
            spin_samples.append(spin_samples_par[0][i])
            error_samples.append(error_samples_par[0][i])
            k_samples.append(len(spin_samples_par[0][i]))
            T2_samples.append(exp_params['T2'])
            count += 1
    
    
    T2_samples = np.vstack(T2_samples)
    
    return k_samples, spin_samples, T2_samples, error_samples      




def fit_exp_data_rjmcmc_fixed_hf_df(HF_FILE, HF_DIST_MAT_FILE, LOWER_THRESH, NOISE_8,
                        NOISE_16, INITIAL_T2, K_MAX, NUM_TRIALS, NUM_STRANDS, BETA, R_SPIN, R_T2, NUM_ENSEMBLES, 
                                    DATA_PATH_8, DATA_PATH_16,
                        NV_NUM, NUM_RJMCMC_STEPS, NUM_T2_STEPS, NUM_PAR_STEPS, SIGMA_SQ):
    
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
    noise_exp_param = [np.sqrt(np.linalg.norm(NOISE_8**2/len(TIME_8))),
             np.sqrt(np.linalg.norm(NOISE_16**2/len(TIME_16)))]
    time = [TIME_8, TIME_16]
    initial_exp_params = rjmcmc.make_exp_params_dict(num_experiments,
                                      num_pulses,
                                      mag_field,
                                      noise_exp_param,
                                      time,
                                      INITIAL_T2)

    coherence_signals = []
    coherence_signals.append(data_8)
    coherence_signals.append(data_16)
    
    # ab initio hf data and bootstrap df
    noise = NOISE_8 + NOISE_16
    
    hf_df = rjmcmc.make_df_from_Ivady_file(HF_FILE, 200, LOWER_THRESH)

    hf_dist_mat = pkl.load(open(HF_DIST_MAT_FILE, 'rb'))
   
    
    ensembles = []
    num_ensembles = 0
    while num_ensembles < NUM_ENSEMBLES:
        print(num_ensembles)
        ensemble_dict = {}
        num_spins_initial = np.random.choice(range(1, K_MAX+1))
        spin_indices = np.arange(len(hf_df))
        spin_list_initial = np.random.choice(spin_indices, size=num_spins_initial, replace=False)
    
        k_trials, spin_trials, T2_trials, error_trials, accept_dict = RJMCMC_RWMH_with_parallel_tempering(spin_list_initial, 
                                                                                               hf_df, hf_dist_mat, R_SPIN,
                                                             R_T2, initial_exp_params, coherence_signals, NUM_TRIALS, 
                                                                      K_MAX, NUM_STRANDS, BETA, NUM_RJMCMC_STEPS, NUM_PAR_STEPS,
                                                                                              NUM_T2_STEPS, SIGMA_SQ)
        ensemble_dict['initial_spins'] = spin_list_initial
        ensemble_dict['k_trials'] = k_trials
        ensemble_dict['spin_trials'] = spin_trials
        ensemble_dict['error_trials'] = error_trials
        ensemble_dict['T2_trials'] = T2_trials
        ensembles.append(ensemble_dict)
        num_ensembles += 1
    
    return ensembles, exp_params, coherence_signals, hf_df
