import matplotlib.pyplot as plt
from collections import Counter
import statistics
from mpl_toolkits.mplot3d import Axes3D
import math

def get_spin_config_lowest_err(ensembles):
    lowest_err = 100
    lowest_ensemble_index = 0
    lowest_error_index = 0

    for i in range(len(ensembles)):
        if min(ensembles[i]['error_trials']) < lowest_err:
            lowest_err = min(ensembles[i]['error_trials'])
            lowest_ensemble_index = i
            lowest_error_index = ensembles[i]['error_trials'].index(lowest_err)
            
    best_spin_fit = ensembles[lowest_ensemble_index]['spin_trials'][lowest_error_index]
    return best_spin_fit

def plot_error_trajectories(saved_data):
    ensembles = saved_data['ensembles']
    for i in range(len(ensembles)):
        plt.plot(ensembles[i]['error_trials'])
    plt.xlabel('steps')
    plt.ylabel('error')
    plt.title(f"{i+1} ensembles from fitting {saved_data['exp_params']['num_experiments']} coherence signals")
    
    
def plot_k_trajectories(saved_data):
    ensembles = saved_data['ensembles']
    for i in range(len(ensembles)):
        plt.plot(ensembles[i]['k_trials'], alpha=0.2)
    plt.xlabel('steps')
    plt.ylabel('number of spin')
    plt.title(f'{i+1} ensembles from fitting {saved_data['exp_params']['num_experiments']} coherence signals') 
    
    
    
def plot_best_fit_coherence_signal(saved_data, index):
    
    best_spin = get_spin_config_lowest_err(saved_data['ensembles'])
    ground_spin = saved_data['spin_list_ground']
    
    ground_coherence = calculate_coherence(ground_spin,
                                           saved_data['hf_df'],
                                           saved_data['exp_params'])
    
    best_fit_coherence = calculate_coherence(best_spin,
                                           saved_data['hf_df'],
                                           saved_data['exp_params'])
    
    exp_params = saved_data['exp_params']
    time = exp_params['timepoints'][index]
    
    plt.plot(time, ground_coherence[index], label = 'ground')
    plt.plot(time, best_fit_coherence[index], ls='--', label = 'best fit')
    plt.xlabel('time (ms)')
    plt.ylabel('coherence')
    plt.legend()
    plt.title(f'best fit from {exp_params['num_experiments']} signals, (XY{exp_params['num_pulses'][index]}), {len(best_spin)} spins')
    
def most_frequent_inner_list(lst_of_lsts):
    # Use Counter to count occurrences of each inner list (converted to tuple)
    count = Counter(tuple(inner_list) for inner_list in lst_of_lsts)
    # Find the most common inner list
    most_common_inner_list, _ = count.most_common(1)[0]
    # Convert back to list
    return list(most_common_inner_list)


def get_posterior_error_thresh(saved_data, error_thresh):
    ensembles = saved_data['ensembles']
    hf_df = saved_data['hf_df']
    
    k_posterior = []
    posterior_data = {}
    
    for i, ensemble in enumerate(ensembles):
        print(i)
        error_trials = ensemble['error_trials']
        k_trials = ensemble['k_trials']
        spin_trials = ensemble['spin_trials']
        
        for j, (error, k) in enumerate(zip(error_trials, k_trials)):
            # if j % 10000 == 0:
            #     print(j)
            if error < error_thresh:
                k_posterior.append(k)
                
                if k not in posterior_data:
                    posterior_data[k] = {
                        'A_perp': [],
                        'A_par': [],
                        'x': [],
                        'y': [],
                        'z': [],
                        'spin': [],
                        'spin_by_group': []
                    }
                
                posterior_data_k = posterior_data[k]
                spins = spin_trials[j]
                posterior_data_k['spin_by_group'].append(spins)
                
                for spin in spins:
                    hf_row = hf_df.iloc[spin]
                    posterior_data_k['A_perp'].append(hf_row['A_perp'])
                    posterior_data_k['A_par'].append(hf_row['A_par'])
                    posterior_data_k['x'].append(hf_row['x'])
                    posterior_data_k['y'].append(hf_row['y'])
                    posterior_data_k['z'].append(hf_row['z'])
                    posterior_data_k['spin'].append(spin)
    
    return k_posterior, posterior_data

def plot_k_histogram(k_posterior, saved_data):
    num_spins = len(saved_data['spin_list_ground'])
    plt.hist(k_posterior, bins=range(min(k_posterior), max(k_posterior) + 2), 
             edgecolor='black', align='left', density=True)
    plt.title(f'mode: {statistics.mode(k_posterior)}, simulated spins: {num_spins}')
    plt.xlabel('number of spins')


def get_top_n_modal_values(lst, n=19):
    # Count the frequency of each element in the list
    counts = Counter(lst)
    
    # Get the most common elements as a list of tuples (element, frequency)
    most_common = counts.most_common(n)
    
    # Extract only the elements from the tuples
    top_n_values = [element for element, frequency in most_common]
    
    return top_n_values
    

def get_modal_spins(k_posterior, posterior_data):
    k_mode = statistics.mode(k_posterior)
    modal_spins = most_frequent_inner_list(posterior_data[k_mode]['spin_by_group'])
    
    return k_mode, modal_spins
    

def plot_modal_fit_coherence(k_posterior, posterior_data, saved_data, index):
    
    ground_spin = saved_data['spin_list_ground']
    _, modal_spins = get_modal_spins(k_posterior, posterior_data)
    
    ground_coherence = calculate_coherence(ground_spin,
                                           saved_data['hf_df'],
                                           saved_data['exp_params'])
    modal_coherence = calculate_coherence(modal_spins,
                                      saved_data['hf_df'],
                                      saved_data['exp_params'])
    exp_params = saved_data['exp_params']
    time = exp_params['timepoints'][index]
    
    plt.plot(time, ground_coherence[index], label = 'ground')
    plt.plot(time, modal_coherence[index], ls='--', label = 'best fit')
    plt.xlabel('time (ms)')
    plt.ylabel('coherence')
    plt.legend()
    plt.title(f'modal fit from {exp_params['num_experiments']} (XY{exp_params['num_pulses'][index]}), {len(modal_spins)} spins')
    
    
def plot_HF_ground(saved_data):
    ground_spins = saved_data['spin_list_ground']
    hf_df = saved_data['hf_df']
    hf_A_perp = []
    hf_A_par = []
    for spin in ground_spins:
        hf_A_perp.append(hf_df.iloc[spin]['A_perp'])
        hf_A_par.append(hf_df.iloc[spin]['A_par'])
        
    plt.hist2d(hf_A_perp, hf_A_par, range=[[0, 200],[-200, 200]], bins=(50,100))
    plt.xlabel('A_perp')
    plt.ylabel('A_par')
    plt.title(f'ground hyperfines ({len(ground_spins)} spins)')
    

def plot_HF_best_fit(saved_data):
    best_spin = get_spin_config_lowest_err(saved_data['ensembles'])
    hf_df = saved_data['hf_df']
    hf_A_perp = []
    hf_A_par = []
    for spin in best_spin:
        hf_A_perp.append(hf_df.iloc[spin]['A_perp'])
        hf_A_par.append(hf_df.iloc[spin]['A_par'])
        
    plt.hist2d(hf_A_perp, hf_A_par, range=[[0, 200],[-200, 200]], bins=(50,100))
    plt.xlabel('A_perp')
    plt.ylabel('A_par')
    plt.title(f'best fit hyperfines ({len(best_spin)} spins)')
    
def plot_HF_modal_fit(saved_data, k_posterior, posterior_data):
    _, modal_spins = get_modal_spins(k_posterior, posterior_data)
    hf_df = saved_data['hf_df']
    hf_A_perp = []
    hf_A_par = []
    for spin in modal_spins:
        hf_A_perp.append(hf_df.iloc[spin]['A_perp'])
        hf_A_par.append(hf_df.iloc[spin]['A_par'])
        
    plt.hist2d(hf_A_perp, hf_A_par, range=[[0, 200],[-200, 200]], bins=(50,100))
    plt.xlabel('A_perp')
    plt.ylabel('A_par')
    plt.title(f'modal fit hyperfines ({len(modal_spins)} spins)')
    

def plot_HF_posterior_k(posterior_data, index):
    
    hf_A_perp = posterior_data[index]['A_perp']
    hf_A_par = posterior_data[index]['A_par']
    
    plt.hist2d(hf_A_perp, hf_A_par, range=[[0, 200],[-200, 200]], bins=(50,100))
    plt.xlabel('A_perp')
    plt.ylabel('A_par')
    plt.title(f'posterior hyperfines for k = {index}')
    
    
def plot_ground_spins_xyz(saved_data):
    
    ground_spins = saved_data['spin_list_ground']
    hf_df = saved_data['hf_df']
    x_ground = []
    y_ground = []
    z_ground = []
    
    A_perp = []
    A_par = []
    
    x_lattice = []
    y_lattice = []
    z_lattice = []
    
    x_symm = []
    y_symm = []
    z_symm = []
    
    for spin in ground_spins:
        hf_row = hf_df.iloc[spin]
        x_ground.append(hf_row['x'])
        y_ground.append(hf_row['y'])
        z_ground.append(hf_row['z'])
        A_perp.append(hf_row['A_perp'])
        A_par.append(hf_row['A_par'])
       
    
    for index, row in hf_df.iterrows():
        x_lattice.append(row['x'])
        y_lattice.append(row['y'])
        z_lattice.append(row['z'])
        if is_pair_close_to_any(row['A_perp'], row['A_par'], A_perp, A_par):
            x_symm.append(row['x'])
            y_symm.append(row['y'])
            z_symm.append(row['z'])
   
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_lattice, y_lattice, z_lattice, color='grey', alpha=0.01)
    ax.scatter(x_symm, y_symm, z_symm, alpha=0.5, color='green', label='symmetry')
    ax.scatter(x_ground, y_ground, z_ground, color='blue', alpha=1, label='ground spins')

    # Set labels
    ax.set_xlabel('X (A)')
    ax.set_ylabel('Y (A)')
    ax.set_zlabel('Z (A)')
    plt.title('ground spins')
    plt.legend()
    plt.show()
    
    
def is_close_to_any(value, values_list, rel_tol=1e-1, abs_tol=0.0):
    return any(math.isclose(value, v, rel_tol=rel_tol, abs_tol=abs_tol) for v in values_list)

def is_pair_close_to_any(x, y, x_list, y_list, rel_tol=1e-3, abs_tol=0.0):
    for x_val, y_val in zip(x_list, y_list):
        if math.isclose(x, x_val, rel_tol=rel_tol, abs_tol=abs_tol) and math.isclose(y, y_val, rel_tol=rel_tol, abs_tol=abs_tol):
            return True
    return False

    
def plot_best_fit_spins_xyz(saved_data):
    
    best_spin = get_spin_config_lowest_err(saved_data['ensembles'])
    hf_df = saved_data['hf_df']
    x_ground = []
    y_ground = []
    z_ground = []
    
    x_lattice = []
    y_lattice = []
    z_lattice = []
    
    for spin in best_spin:
        hf_row = hf_df.iloc[spin]
        x_ground.append(hf_row['x'])
        y_ground.append(hf_row['y'])
        z_ground.append(hf_row['z'])
    
    for index, row in hf_df.iterrows():
        x_lattice.append(row['x'])
        y_lattice.append(row['y'])
        z_lattice.append(row['z'])
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_lattice, y_lattice, z_lattice, color='grey', alpha=0.01)
    ax.scatter(x_ground, y_ground, z_ground, alpha=1, label='ground spins')

    # Set labels
    ax.set_xlabel('X (A)')
    ax.set_ylabel('Y (A)')
    ax.set_zlabel('Z (A)')
    plt.title(f'best fit ({len(best_spin)} spins)')
    plt.show()

def plot_modal_fit_spins_xyz(saved_data, k_posterior, posterior_data):
    
    _, modal_spins = get_modal_spins(k_posterior, posterior_data)
    hf_df = saved_data['hf_df']
    x_ground = []
    y_ground = []
    z_ground = []
    
    x_lattice = []
    y_lattice = []
    z_lattice = []
    
    for spin in modal_spins:
        hf_row = hf_df.iloc[spin]
        x_ground.append(hf_row['x'])
        y_ground.append(hf_row['y'])
        z_ground.append(hf_row['z'])
    
    for index, row in hf_df.iterrows():
        x_lattice.append(row['x'])
        y_lattice.append(row['y'])
        z_lattice.append(row['z'])
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_lattice, y_lattice, z_lattice, color='grey', alpha=0.01)
    ax.scatter(x_ground, y_ground, z_ground, alpha=1, label='ground spins')

    # Set labels
    ax.set_xlabel('X (A)')
    ax.set_ylabel('Y (A)')
    ax.set_zlabel('Z (A)')
    plt.title(f'modal fit ({len(modal_spins)} spins)')
    plt.show()
    
    
def plot_posterior_spins_k_xyz(saved_data, posterior_data, index):
    
    
    hf_df = saved_data['hf_df']
    
    x_lattice = []
    y_lattice = []
    z_lattice = []
    
    for i, row in hf_df.iterrows():
        x_lattice.append(row['x'])
        y_lattice.append(row['y'])
        z_lattice.append(row['z'])
    
    x_posterior = posterior_data[index]['x']
    y_posterior = posterior_data[index]['y']
    z_posterior = posterior_data[index]['z']
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_lattice, y_lattice, z_lattice, color='grey', alpha=0.01)
    ax.scatter(x_posterior, y_posterior, z_posterior, color='blue', alpha=1)

    # Set labels
    ax.set_xlabel('X (A)')
    ax.set_ylabel('Y (A)')
    ax.set_zlabel('Z (A)')
    plt.title(f'posterior for k = {index}')
    plt.show()
        
    
