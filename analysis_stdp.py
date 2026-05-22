import numpy as np
import matplotlib.pyplot as plt
import json
import os
import math
import sys
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--path", type=str, default='data/', help='Path to the data directory (default: data/).')
args = parser.parse_args()

data_path = os.path.join(os.getcwd(), args.path+"/")

# Load network and simulation parameters from JSON files
with open(data_path+'network_params.json', 'r') as f:
    network_params = json.load(f)
    
with open(data_path+'simulation_params.json', 'r') as f:
    simulation_params = json.load(f)

def load_spike_data(overlap = False):
    n_spike_dats = network_params["p"]
    # Checks if there is the spike data for the non selective pop
    if os.path.isfile(data_path + "spikedata5.dat"):
        n_spike_dats += 1
    
    if(overlap==False):
        srs = [np.loadtxt(data_path + "spikedata"+str(i)+".dat") for i in range(n_spike_dats)]
        
    # change neuron id so that each selective population has ids [800*(i-1), 800*i]
    else:
        sr0_dum = np.loadtxt(data_path + "spikedata0.dat")
        sr1_dum = np.loadtxt(data_path + "spikedata1.dat")
        sr2_dum = np.loadtxt(data_path + "spikedata2.dat")
        sr3_dum = np.loadtxt(data_path + "spikedata3.dat")
        sr4_dum = np.loadtxt(data_path + "spikedata4.dat")
        old_data = [sr0_dum, sr1_dum, sr2_dum, sr3_dum, sr4_dum]
        ids = np.loadtxt(data_path + "selective_pop_ids.dat")
        srs = []
        for i in range(len(old_data)):
            sorted_ids = np.sort(ids[:,i]) + 1
            sr = old_data[i]
            for idx in sorted_ids:
                pos_old_ids = np.where(old_data[i][:,0]==idx)
                if(list(pos_old_ids[0])!=[]):
                    for r in list(pos_old_ids[0]):
                        sr[r,0]=np.where(sorted_ids==idx)[0][0] + 800*i
            srs.append(sr)
    
    return(srs)

def load_synaptic_weights(filepath: str) -> dict:
    """
    Reads a .dat file containing synaptic weights and organizes them into a dictionary.
    
    Args:
        filepath (str): The path to the file to be read (e.g., 'weights_6000.0.dat').
        
    Returns:
        dict: A dictionary where the keys are the connection names 
              (e.g., 'weights_selective_pop0_to_selective_pop1') and the values 
              are numpy arrays containing the weights (float).
    """
    weights_dict = {}
    current_connection = None

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            
            if not line:
                continue

            if line.startswith("weights_"):
                current_connection = line
                weights_dict[current_connection] = []
            
            elif current_connection is not None:
                pesi = [float(x) for x in line.split()]
                weights_dict[current_connection].extend(pesi)

    for connection in weights_dict:
        weights_dict[connection] = np.array(weights_dict[connection])

    return weights_dict

def raster_plot(data_path):
    labelsize=19
    titlesize=20
    f = network_params["f"]
    N_E = network_params["N_exc"]
    # fraction of neurons belonging to a selective population that we want to plot
    frac_sel_pop = 1.0
    # number of neurons belonging to a selective populaiton
    n_E = int(N_E * f)
    n_E_frac = int(n_E * frac_sel_pop) # prima era n_E * 0.1
    colors = ["blue", "red", "green", "orange", "olive", "cornflowerblue", "salmon", "lime", "gold", "yellowgreen"]
    fig, ax = plt.subplots(figsize=(15,10))
    for i in range(len(srs)):
        dum = srs[i]
        dumx = []
        dumy = []
        for j in range(len(dum[:,0])):
            if(srs[i][j,0]<n_E_frac+n_E*i+2):
                dumx.append(dum[j,1])
                dumy.append(dum[j,0]-(n_E-n_E_frac)*i - 2)
        ax.plot(dumx, dumy, '.', color=colors[i])
    ax.set_ylabel("# cell", fontsize=labelsize)
    ax.set_xlabel("Time [ms]", fontsize=labelsize)
    ax.set_xlim(0,20000)
    ax.set_ylim(0,n_E_frac*len(srs))
    ax.tick_params(labelsize=labelsize)
    for i in range(network_params["item_loading"]["nstim"]):
        ax.axvspan(network_params["item_loading"]["origin"][i], network_params["item_loading"]["origin"][i]+network_params["stimulation_params"]["T_cue"], (1./len(srs))*i, (1./len(srs))*(i+1), alpha=0.5, color='grey')
    if("nonspecific_readout_signals" in network_params):
        for i in range(network_params["nonspecific_readout_signals"]["nstim"]):
            if(i==0):
                ax.axvspan(network_params["nonspecific_readout_signals"]["origin"][i], network_params["nonspecific_readout_signals"]["origin"][i]+network_params["stimulation_params"]["T_reac"], alpha=0.5, color='cornflowerblue', label="Readout signal")
            else:
                ax.axvspan(network_params["nonspecific_readout_signals"]["origin"][i], network_params["nonspecific_readout_signals"]["origin"][i]+network_params["stimulation_params"]["T_reac"], alpha=0.5, color='cornflowerblue')
    if("nonspecific_noise" in network_params):
        for i in range(network_params["nonspecific_noise"]["nstim"]):
            if(i==0):
                ax.axvspan(network_params["nonspecific_noise"]["origin"][i], network_params["nonspecific_noise"]["origin"][i]+network_params["stimulation_params"]["T_reac"], alpha=0.5, color='turquoise', label="Noise")
            else:
                ax.axvspan(network_params["nonspecific_noise"]["origin"][i], network_params["nonspecific_noise"]["origin"][i]+network_params["stimulation_params"]["T_reac"], alpha=0.5, color='turquoise')
    plt.subplots_adjust(left=0.07, right=0.976, top=0.925, bottom=0.1)
    #plt.savefig(simulation_params['data_path']+"raster_plot_analysis.png")
    plt.savefig(data_path+"raster_plot_analysis.png")
    plt.draw()


def instantaneus_firing_rate(sr, binwidth = 25):
    # Calculate the instantaneous firing rate for each selective population 
    # using a histogram with a specified bin width (in ms)

    # the time of the istantaneous firing rate is the center of each bin

    # this function returns the time and the firing rate for each selective population

    
    
    # selective population 0
    if(np.size(sr[0]) > 2):
        fr0 = sr[0][:,1]
        frmax0 = np.max(np.abs(fr0))
        lim0 = (int(frmax0/binwidth) + 1) * binwidth
        bins0 = np.arange(0, lim0 + binwidth, binwidth)

        h0 = np.histogram(fr0, bins=bins0)[0:2]
        # frequency in Hz per bin, nomalized
        fr0 = (h0[0]/(binwidth/1000.0))/(network_params["N_exc"]*network_params["f"])
        # time of the center of each bin
        time0 = [(h0[1][i]+h0[1][i+1])/2.0 for i in range(len(h0[0]))]
    else:
        time0 = []
        fr0 = []

    # selective population 1
    if(np.size(sr[1]) > 2):
        fr1 = sr[1][:,1]
        frmax1 = np.max(np.abs(fr1))
        lim1 = (int(frmax1/binwidth) + 1) * binwidth
        bins1 = np.arange(0, lim1 + binwidth, binwidth)

        h1 = np.histogram(fr1, bins=bins1)[0:2]
        # frequency in Hz per bin, nomalized
        fr1 = (h1[0]/(binwidth/1000.0))/(network_params["N_exc"]*network_params["f"])
        # time of the center of each bin
        time1 = [(h1[1][i]+h1[1][i+1])/2.0 for i in range(len(h1[0]))]
    else:
        time1 = []
        fr1 = []
    
    # selective population 2
    if(np.size(sr[2]) > 2):
        fr2 = sr[2][:,1]
        frmax2 = np.max(np.abs(fr2))
        lim2 = (int(frmax2/binwidth) + 1) * binwidth
        bins2 = np.arange(0, lim2 + binwidth, binwidth)

        h2 = np.histogram(fr2, bins=bins2)[0:2]
        # frequency in Hz per bin, nomalized
        fr2 = (h2[0]/(binwidth/1000.0))/(network_params["N_exc"]*network_params["f"])
        # time of the center of each bin
        time2 = [(h2[1][i]+h2[1][i+1])/2.0 for i in range(len(h2[0]))]
    else:
        time2 = []
        fr2 = []

    # selective population 3
    if(np.size(sr[3]) > 2):
        fr3 = sr[3][:,1]
        frmax3 = np.max(np.abs(fr3))
        lim3 = (int(frmax3/binwidth) + 1) * binwidth
        bins3 = np.arange(0, lim3 + binwidth, binwidth)

        h3 = np.histogram(fr3, bins=bins3)[0:2]
        # frequency in Hz per bin, nomalized
        fr3 = (h3[0]/(binwidth/1000.0))/(network_params["N_exc"]*network_params["f"])
        # time of the center of each bin
        time3 = [(h3[1][i]+h3[1][i+1])/2.0 for i in range(len(h3[0]))]
    else:
        time3 = []
        fr3 = []

    # selective population 4
    if(np.size(sr[4]) > 2):
        fr4 = sr[4][:,1]
        frmax4 = np.max(np.abs(fr4))
        lim4 = (int(frmax4/binwidth) + 1) * binwidth
        bins4 = np.arange(0, lim4 + binwidth, binwidth)
        
        h4 = np.histogram(fr4, bins=bins4)[0:2]
        # frequency in Hz per bin, nomalized
        fr4 = (h4[0]/(binwidth/1000.0))/(network_params["N_exc"]*network_params["f"])
        # time of the center of each bin
        time4 = [(h4[1][i]+h4[1][i+1])/2.0 for i in range(len(h4[0]))]
    else:
        time4 = []
        fr4 = []
    
    # non selective population
    if(np.size(sr[5]) > 2):
        fr5 = sr[5][:,1]
        frmax5 = np.max(np.abs(fr5))
        lim5 = (int(frmax5/binwidth) + 1) * binwidth
        bins5 = np.arange(0, lim5 + binwidth, binwidth)
        
        h5 = np.histogram(fr5, bins=bins5)[0:2]
        # frequency in Hz per bin, nomalized
        fr5 = (h5[0]/(binwidth/1000.0))/(network_params["N_exc"]*network_params["f"])
        # time of the center of each bin
        time5 = [(h5[1][i]+h5[1][i+1])/2.0 for i in range(len(h5[0]))]
    else:
        time5 = []
        fr5 = []

    return time0, fr0, time1, fr1, time2, fr2, time3, fr3, time4, fr4, time5, fr5

def plot_instantaneus_firing_rate(sr, data_path=data_path):
    labelsize=19
    titlesize=20

    time0, fr0, time1, fr1, time2, fr2, time3, fr3, time4, fr4, time5, fr5 = instantaneus_firing_rate(sr, binwidth = 25)

    fig, ax = plt.subplots(figsize=(15,10))
    if(time0!=[]):
        ax.plot(time0, fr0, color='blue', label="Selective population 0")
    else:
        ax.axhline(0, color='blue', label="Selective population 0")
    if(time1!=[]):
        ax.plot(time1, fr1, color='red', label="Selective population 1")
    else:
        ax.axhline(0, color='red', label="Selective population 1")
    if(time2!=[]):
        ax.plot(time2, fr2, color='green', label="Selective population 2")
    else:
        ax.axhline(0, color='green', label="Selective population 2")
    if(time3!=[]):
        ax.plot(time3, fr3, color='orange', label="Selective population 3")
    else:
        ax.axhline(0, color='orange', label="Selective population 3")
    if(time4!=[]):
        ax.plot(time4, fr4, color='olive', label="Selective population 4")
    else:
        ax.axhline(0, color='olive', label="Selective population 4")
    if(time5!=[]):
        ax.plot(time5, fr5, color='purple', label="Non-selective population")
    else:
        ax.axhline(0, color='purple', label="Non-selective population")

    ax.set_ylabel("Firing rate [Hz]", fontsize=labelsize)
    ax.set_xlabel("Time [ms]", fontsize=labelsize)
    ax.set_xlim(0, simulation_params["t_sim"]+500)
    ax.tick_params(labelsize=labelsize)
    ax.legend(fontsize=labelsize)

    plt.savefig(data_path+"instantaneous_firing_rate.png")
    plt.draw()

def firing_rate(sr, start_time, stop_time, data_path=data_path):
    # Calculate the firing rate for each neuron in the time window [start_time, stop_time]

    # select the spikes that are in the time window [start_time, stop_time]
    time_mask = (sr[:,1] >= start_time) & (sr[:,1] <= stop_time)
    spike_times = sr[:,1][time_mask]
    id_neurons = sr[:,0][time_mask]

    # Calculate the firing rate for each neuron using numpy's unique function
    # it returns the unique neuron id and the number of occurence of every unique neuron id
    id_neur, counts = np.unique(id_neurons, return_counts=True)
    firing_rate = counts / ((stop_time - start_time) / 1000.0)

    return firing_rate

def plot_firing_rate_histogram(firing_rates_dict, data_path=None, filename = ""):
    
    num_pops = len(firing_rates_dict)
    if num_pops == 0:
        print("The firing rates dictionary is empty.")
        return

    cols = 2  # Set how many columns you want side-by-side
    rows = math.ceil(num_pops / cols) # Calculate the required rows automatically

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

    # Ensure 'axes' is a 1D (flat) array for easy iteration, 
    # even if there's only one row or a single plot
    if num_pops == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (population, firing_rates) in zip(axes, firing_rates_dict.items()):
        
        # Draw the histogram in the specific subplot
        ax.hist(firing_rates, bins="auto", color='steelblue', edgecolor='black', alpha=0.7)
        
        # Labels and Titles
        ax.set_title(f'Firing Rate - {population}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Firing Rate (Hz)')
        ax.set_ylabel('Count (Neurons)')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    # hide the remaining empty subplots
    for i in range(num_pops, len(axes)):
        fig.delaxes(axes[i])

    # Optimize spacing to prevent label overlapping
    plt.tight_layout()
    
    if data_path:
        plt.savefig(f"{data_path}/" + filename + ".png", dpi=300)
    
    # Display the grid on screen
    # plt.show()

def plot_weights_histogram(weights_dict, data_path="", num = ""):
    labelsize = 14
    titlesize = 16
    
    num_plots = len(weights_dict)
    
    if num_plots == 0:
        print("No data to plot.")
        return

    cols = 3
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    
    if num_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, (connection, weights) in enumerate(weights_dict.items()):
        ax = axes[i]
        
        ax.hist(weights, bins="auto", alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(17.04, color='blue', linestyle='dashed', linewidth=1.5, label="baseline weight")
        ax.axvline(76.7, color='red', linestyle='dashed', linewidth=1.5, label="potentiated weight")
        
        ax.set_title(connection, fontsize=titlesize)
        ax.set_xlabel("Synaptic weight", fontsize=labelsize)
        ax.set_ylabel("Count", fontsize=labelsize)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='upper right')

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    plt.savefig(data_path + "weights_histogram_grid" + num + ".png")
    plt.draw()

overlap = network_params["overlap"]
srs = load_spike_data(overlap = overlap)

sr0 = srs[0]
sr1 = srs[1]
sr2 = srs[2]
sr3 = srs[3]
sr4 = srs[4]
sr5 = srs[5] if len(srs) > 5 else None

weight_dict_1 = load_synaptic_weights(data_path + "weights_9000.0.dat")

plot_weights_histogram(weight_dict_1, data_path, num="_1")


# print("weights from selective population 1 to selective population 2: ", weight_dict["weights_selective_pop0_to_selective_pop1"][0])

# print("data path: ", data_path)

# raster_plot(data_path)

# plt.show()

start_time_after = network_params["item_loading"]["origin"][0] + network_params["stimulation_params"]["T_cue"]
stop_time_after = simulation_params["t_sim"]

start_time_before = 0.0
stop_time_before = network_params["item_loading"]["origin"][0]

plot_instantaneus_firing_rate(srs)

firing_rates_dict_after = {
    "Selective population 0": firing_rate(sr0, start_time=start_time_after, stop_time=stop_time_after),
    "Selective population 1": firing_rate(sr1, start_time=start_time_after, stop_time=stop_time_after),
    "Selective population 2": firing_rate(sr2, start_time=start_time_after, stop_time=stop_time_after),
    "Selective population 3": firing_rate(sr3, start_time=start_time_after, stop_time=stop_time_after),
    "Selective population 4": firing_rate(sr4, start_time=start_time_after, stop_time=stop_time_after),
    "Non selective population": firing_rate(sr5, start_time=start_time_after, stop_time=stop_time_after) if sr5 is not None else []
}

plot_firing_rate_histogram(firing_rates_dict_after, data_path, filename="firing_rate_after")

firing_rates_dict_before = {
    "Selective population 0": firing_rate(sr0, start_time=start_time_before, stop_time=stop_time_before),
    "Selective population 1": firing_rate(sr1, start_time=start_time_before, stop_time=stop_time_before),
    "Selective population 2": firing_rate(sr2, start_time=start_time_before, stop_time=stop_time_before),
    "Selective population 3": firing_rate(sr3, start_time=start_time_before, stop_time=stop_time_before),  
    "Selective population 4": firing_rate(sr4, start_time=start_time_before, stop_time=stop_time_before),
    "Non selective population": firing_rate(sr5, start_time=start_time_before, stop_time=stop_time_before) if sr5 is not None else []
}

plot_firing_rate_histogram(firing_rates_dict_before, data_path, filename="firing_rate_before")

# plt.show()