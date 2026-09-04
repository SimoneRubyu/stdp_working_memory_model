import numpy as np
import matplotlib.pyplot as plt
import json
import os
import math
import sys
import pandas as pd
from argparse import ArgumentParser
from scipy.stats import skew, kurtosis

parser = ArgumentParser()
parser.add_argument("--path", type=str, default='data', help='Path to the data directory (default: data/).')
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
    if os.path.isfile(data_path + "spikedata6.dat"):
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
        ax.axvspan(network_params["item_loading"]["origin"][i], network_params["item_loading"]["origin"][i]+network_params["item_loading"]["stop"][i], (1./len(srs))*i, (1./len(srs))*(i+1), alpha=0.5, color='grey')
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
    if(len(sr) > 5 and np.size(sr[5]) > 2):
        fr5 = sr[5][:,1]
        frmax5 = np.max(np.abs(fr5))
        lim5 = (int(frmax5/binwidth) + 1) * binwidth
        bins5 = np.arange(0, lim5 + binwidth, binwidth)
        
        h5 = np.histogram(fr5, bins=bins5)[0:2]
        # frequency in Hz per bin, nomalized
        fr5 = (h5[0]/(binwidth/1000.0))/(network_params["N_exc"]*(1-network_params["f"]*5))
        # time of the center of each bin
        time5 = [(h5[1][i]+h5[1][i+1])/2.0 for i in range(len(h5[0]))]
    else:
        time5 = []
        fr5 = []
    
    # inhibitory population
    if(len(sr) > 6 and np.size(sr[6]) > 2):
        fr6 = sr[6][:,1]
        frmax6 = np.max(np.abs(fr6))
        lim6 = (int(frmax6/binwidth) + 1) * binwidth
        bins6 = np.arange(0, lim6 + binwidth, binwidth)
        
        h6 = np.histogram(fr6, bins=bins6)[0:2]
        # frequency in Hz per bin, nomalized
        fr6 = (h6[0]/(binwidth/1000.0))/(network_params["N_inh"])
        # time of the center of each bin
        time6 = [(h6[1][i]+h6[1][i+1])/2.0 for i in range(len(h6[0]))]
    else:
        time6 = []
        fr6 = []

    return time0, fr0, time1, fr1, time2, fr2, time3, fr3, time4, fr4, time5, fr5, time6, fr6

def plot_instantaneus_firing_rate(sr, data_path=data_path):
    labelsize=19
    titlesize=20

    time0, fr0, time1, fr1, time2, fr2, time3, fr3, time4, fr4, time5, fr5, time6, fr6 = instantaneus_firing_rate(sr, binwidth = 25)

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
    if(time6!=[]):
        ax.plot(time6, fr6, color='black', label="Inhibitory population")
    else:
        ax.axhline(0, color='black', label="Inhibitory population")
    ax.set_ylabel("Firing rate [Hz]", fontsize=labelsize)
    ax.set_xlabel("Time [ms]", fontsize=labelsize)
    ax.set_xlim(0, simulation_params["t_sim"]+500)
    ax.tick_params(labelsize=labelsize)
    ax.legend(fontsize=labelsize)

    plt.savefig(data_path+"instantaneous_firing_rate.png")
    plt.draw()

def firing_rate(t_start, t_stop, sr, id_min=None, n_neurons=None):
    """
    Firing rate (in Hz) of every neuron of one population, in the window
    [t_start, t_stop).

    Parameters
    ----------
    t_start, t_stop : float
        Window boundaries, in ms.
    sr : (n_spikes, 2) array
        Spike recorder data: column 0 = neuron id, column 1 = spike time [ms].
    id_min : int, optional
        Lowest neuron id of the population. Defaults to the smallest id
        appearing anywhere in `sr`.
    n_neurons : int, optional
        Size of the population. Defaults to (max id - min id + 1) over the
        whole recording. Pass it explicitly when some neurons at the edges of
        the id range never fire at all: they cannot be inferred from the data.

    Returns
    -------
    ids : (n_neurons,) int array
        Neuron ids.
    rates : (n_neurons,) float array
        One firing rate per neuron, in Hz. Neurons that are silent inside the
        window get 0.0 (they are NOT dropped).
    """
    duration_s = (t_stop - t_start) / 1000.0
    if duration_s <= 0:
        raise ValueError("t_stop must be strictly greater than t_start")

    if sr is None or np.size(sr) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    sr = np.atleast_2d(np.asarray(sr, dtype=float))
    all_senders = np.rint(sr[:, 0]).astype(np.int64)
    times = sr[:, 1]

    # The id range must come from the WHOLE recording, not from the window:
    # otherwise a neuron that happens to be silent inside the window shifts
    # the whole id axis.
    if id_min is None:
        id_min = int(all_senders.min())
    if n_neurons is None:
        n_neurons = int(all_senders.max() - id_min + 1)
    ids = np.arange(id_min, id_min + n_neurons, dtype=np.int64)

    mask = (times >= t_start) & (times < t_stop)
    senders = all_senders[mask]

    # vectorised spike count per neuron; silent neurons stay at 0
    counts = np.zeros(n_neurons, dtype=np.int64)
    if senders.size:
        idx = senders - id_min
        keep = (idx >= 0) & (idx < n_neurons)
        if not keep.all():
            print("  WARNING: {} spikes fall outside the id range "
                  "[{}, {}] and were discarded. Check id_min / n_neurons."
                  .format(int((~keep).sum()), id_min, id_min + n_neurons - 1))
        counts += np.bincount(idx[keep], minlength=n_neurons)

    rates = counts / duration_s

    print("Firing rate on [{}, {}) ms | {} neurons | mean {:.2f} Hz | "
          "median {:.2f} Hz | max {:.2f} Hz | silent {}".format(
              t_start, t_stop, n_neurons, rates.mean(), np.median(rates),
              rates.max(), int((rates == 0).sum())))

    return ids, rates
"""
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
"""
def plot_firing_rate_histogram(firing_rates_dict, data_path=None, filename = "",
                               duration_ms=None, bin_spikes=None, max_bins=40):
    """
    duration_ms : float, optional
        Length of the window used in firing_rate(), i.e. t_stop - t_start.
        A firing rate can only be a multiple of 1000/duration_ms Hz, so when
        this is given the bins are aligned to that lattice and there are no
        spurious empty bins. Falls back to bins="auto" when omitted.
    bin_spikes : int, optional
        Bin width expressed in spike counts (1 = one bin per possible rate).
        Chosen automatically so that no histogram exceeds `max_bins` bins.
    """
    # --- sanitise the input -------------------------------------------------
    # matplotlib's hist() treats a 2D input as SEVERAL datasets (one per
    # column), so passing [[id, rate], ...] silently plots the neuron ids as
    # well. Keep only the rates, and drop the empty populations.
    clean = {}
    for population, firing_rates in firing_rates_dict.items():
        fr = np.asarray(firing_rates, dtype=float)
        if fr.ndim == 2:
            if fr.shape[1] != 2:
                print(f"Skipping '{population}': unexpected shape {fr.shape}.")
                continue
            fr = fr[:, 1]          # second column = rates
        fr = fr.ravel()
        fr = fr[np.isfinite(fr)]
        if fr.size == 0:
            print(f"Skipping '{population}': no data.")
            continue
        clean[population] = fr
 
    num_pops = len(clean)
    if num_pops == 0:
        print("The firing rates dictionary is empty.")
        return
 
    cols = 2  # Set how many columns you want side-by-side
    rows = math.ceil(num_pops / cols) # Calculate the required rows automatically
 
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()
 
    stats = {}
 
    for ax, (population, firing_rates) in zip(axes, clean.items()):
 
        # --- bin edges ------------------------------------------------------
        # A rate is n_spikes * 1000 / duration_ms, so only multiples of
        # `quantum` can occur. bins="auto" ignores this and picks a width that
        # is not a multiple of the quantum, leaving every other bin empty (the
        # comb-like gaps). Aligning the edges to the lattice removes them.
        if duration_ms:
            quantum = 1000.0 / duration_ms
            k_max = int(round(firing_rates.max() / quantum))
            step = bin_spikes if bin_spikes else max(1, math.ceil((k_max + 1) / max_bins))
            edges = np.arange(-0.5, k_max + step, step) * quantum
        else:
            edges = "auto"
 
        # Draw the histogram in the specific subplot
        ax.hist(firing_rates, bins=edges, color='steelblue', edgecolor='black', alpha=0.7)
 
        # --- summary statistics ---------------------------------------------
        mean_fr = float(firing_rates.mean())
        median_fr = float(np.median(firing_rates))
        stats[population] = {
            "n_neurons": int(firing_rates.size),
            "mean": mean_fr,
            "median": median_fr,
            "std": float(firing_rates.std(ddof=1)) if firing_rates.size > 1 else 0.0,
            "cv": float(firing_rates.std(ddof=1) / mean_fr) if (firing_rates.size > 1 and mean_fr > 0) else np.nan,
            "q25": float(np.percentile(firing_rates, 25)),
            "q75": float(np.percentile(firing_rates, 75)),
            "max": float(firing_rates.max()),
            "frac_silent": float((firing_rates == 0).mean()),
        }
 
        ax.axvline(mean_fr, color='red', linestyle='dashed', linewidth=1.5,
                   label=f"mean = {mean_fr:.2f} Hz")
        ax.axvline(median_fr, color='darkgreen', linestyle='dashdot', linewidth=1.5,
                   label=f"median = {median_fr:.2f} Hz")
        
        # Labels and Titles
        ax.set_title(f'Firing Rate - {population}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Firing Rate (Hz)')
        ax.set_ylabel('Count (Neurons)')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)
 
    # hide the remaining empty subplots
    for i in range(num_pops, len(axes)):
        fig.delaxes(axes[i])
 
    # Optimize spacing to prevent label overlapping
    plt.tight_layout()
    
    if data_path:
        plt.savefig(f"{data_path}/" + filename + ".png", dpi=300)
 
    # Summary table on stdout, and the same numbers returned to the caller
    print("\n{:<28} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "population", "N", "mean", "median", "std", "max", "silent"))
    for population, s in stats.items():
        print("{:<28} {:>6d} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f} {:>7.1f}%".format(
            population[:28], s["n_neurons"], s["mean"], s["median"],
            s["std"], s["max"], 100 * s["frac_silent"]))
 
    return stats
    
    # Display the grid on screen
    # plt.show()

def plot_weights_histogram_combined(weights_dict, data_path="", num=""):
    labelsize = 19
    titlesize = 19
    
    if not weights_dict:
        print("No data to plot.")
        return

    all_weights = []
    for weights in weights_dict.values():
        all_weights.extend(weights)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_weights, bins="auto", alpha=0.7, color='steelblue', label='All weights')
        
    #ax.axvline(17, color='blue', linestyle='dashed', linewidth=1.5, label="baseline weight")
    #ax.axvline(77, color='red', linestyle='dashed', linewidth=1.5, label="potentiated weight")
    
    #ax.set_title("Global Synaptic Weights Distribution", fontsize=titlesize)
    ax.set_xlabel("Synaptic weight (pA)", fontsize=labelsize)
    ax.set_ylabel("Count", fontsize=labelsize)
    ax.tick_params(axis='both', labelsize=19)
    #ax.legend(loc='upper right')
    #ax.set_yscale('log')
    ax.set_xlim(0, 40)

    plt.tight_layout()
    plt.savefig(data_path + "weights_histogram_total" + num + ".png")
    #plt.show()

def plot_weights_histogram(weights_dict, data_path="", num = "", nbins=50,
                           w_baseline=17, w_potentiated=77,
                           xlim=None, sharey=False, density=False):
    """
    One histogram per connection, all drawn on a COMMON x axis.

    Parameters
    ----------
    nbins : int
        Number of bins, shared by every panel.
    w_baseline, w_potentiated : float
        Reference weights marked by the vertical dashed lines.
    xlim : (float, float), optional
        Force the x range. By default it is the range of the pooled weights,
        widened so that both reference lines are always visible.
    sharey : bool
        Also put every panel on a common y axis. Useful only when the
        connections have a comparable number of synapses.
    density : bool
        Plot the normalised distribution instead of raw counts, so that
        connections with different synapse counts can be compared by shape.
    """
    labelsize = 14
    titlesize = 16

    num_plots = len(weights_dict)

    if num_plots == 0:
        print("No data to plot.")
        return

    # --- common x axis and common bin edges ---------------------------------
    # Sharing the limits is not enough: with bins="auto" every panel gets its
    # own bin width, so two bars of equal height would represent different
    # amounts of synapses. The bin edges have to be identical too.
    cleaned = {}
    for connection, weights in weights_dict.items():
        w = np.asarray(weights, dtype=float).ravel()
        w = w[np.isfinite(w)]
        if w.size == 0:
            print(f"Skipping '{connection}': no data.")
            continue
        cleaned[connection] = w

    num_plots = len(cleaned)
    if num_plots == 0:
        print("No data to plot.")
        return

    pooled = np.concatenate(list(cleaned.values()))

    if xlim is not None:
        lo, hi = xlim
    else:
        lo = min(pooled.min(), w_baseline, w_potentiated)
        hi = max(pooled.max(), w_baseline, w_potentiated)
        margin = 0.02 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - margin, hi + margin

    edges = np.linspace(lo, hi, nbins + 1)

    cols = 3
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    ymax = 0.0

    for i, (connection, weights) in enumerate(cleaned.items()):
        ax = axes[i]

        counts, _, _ = ax.hist(weights, bins=edges, density=density,
                               alpha=0.7, color='steelblue', edgecolor='black')
        ymax = max(ymax, counts.max())

        ax.axvline(w_baseline, color='blue', linestyle='dashed', linewidth=1.5, label="baseline weight")
        ax.axvline(w_potentiated, color='red', linestyle='dashed', linewidth=1.5, label="potentiated weight")

        ax.set_xlim(lo, hi)          # identical on every panel
        ax.set_title(f"{connection}", fontsize=titlesize)
        ax.set_xlabel("Synaptic weight (pA)", fontsize=labelsize)
        ax.set_ylabel("Density" if density else "Count", fontsize=labelsize)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc='upper right')

    if sharey:
        for ax in axes[:num_plots]:
            ax.set_ylim(0, ymax * 1.05)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    plt.savefig(data_path + "weights_histogram_grid" + num + ".png")
    plt.draw()


def group_weights(weights_dict, populations=None):
    """
    Raggruppa i pesi per popolazione bersaglio: da una parte le connessioni
    ricorrenti, dall'altra tutte quelle in arrivo dalle altre popolazioni.
    """
    if weights_dict is None:
        return None

    if populations is None:
        populations = ([f'selective_pop{i}' for i in range(network_params["p"])]
                       + ['nonselective_pop'])

    grouped = {}
    for target_pop in populations:
        recurrent_key = f'weights_{target_pop}_to_{target_pop}'
        incoming_keys = [k for k in weights_dict.keys()
                         if k.endswith(f'_to_{target_pop}') and k != recurrent_key]

        recurrent_data = np.asarray(weights_dict.get(recurrent_key, np.array([])),
                                    dtype=float)
        incoming_list = [np.asarray(weights_dict[k], dtype=float)
                         for k in incoming_keys if k in weights_dict]
        incoming_data = (np.concatenate(incoming_list) if incoming_list
                         else np.array([]))

        grouped[target_pop] = {'recurrent': recurrent_data,
                               'incoming': incoming_data}
    return grouped


def plot_weights_histogram_grouped(grouped_weights_dict, data_path="", num="", nbins=50,
                                   initial_weights_dict=None,
                                   xlim=None, sharey=False, density=False,
                                   scale_y_on_final=False, logy=False,
                                   label_final="Pesi finali",
                                   label_initial="Pesi iniziali (t = 0)"):
    """
    Un istogramma per connessione, organizzato in 2 colonne (Ricorrenti vs In
    Ingresso) per ogni popolazione, tutti disegnati su un asse X COMUNE.

    Parametri
    ---------
    initial_weights_dict : dict o None
        Stessa struttura di grouped_weights_dict, con i pesi a t = 0 (da
        weights_0.dat). Se dato, in ogni pannello viene sovrapposta in grigio
        la distribuzione iniziale, con gli stessi bin di quella finale.
    scale_y_on_final : bool
        Se i pesi partono tutti dallo stesso valore, l'istogramma iniziale e'
        un picco altissimo in un solo bin e schiaccia la distribuzione finale.
        Con True l'asse y viene scalato sulla sola distribuzione finale: il
        picco iniziale esce dal grafico ma resta visibile dove cade.
    logy : bool
        Asse y logaritmico, alternativa a scale_y_on_final per vedere
        entrambe le distribuzioni senza tagliare nulla.
    """
    labelsize = 22
    titlesize = 22

    num_pops = len(grouped_weights_dict)

    if num_pops == 0:
        print("No data to plot.")
        return

    # --- 1. Pulizia dei dati e pooling per gli assi comuni ---
    def _clean(arr):
        w = np.asarray(arr, dtype=float).ravel()
        return w[np.isfinite(w)]

    cleaned = {}
    cleaned_init = {}
    pooled_list = []

    for pop, data in grouped_weights_dict.items():
        w_rec = _clean(data['recurrent'])
        w_inc = _clean(data['incoming'])
        cleaned[pop] = {'recurrent': w_rec, 'incoming': w_inc}
        # -----------------------------------------------
        # Calcolo mean, std, skewness e kurtosis per ogni popolazione
        mean_rec = np.mean(w_rec)
        print(f"\nMean for {pop} recurrent weights: {mean_rec:.4f}\n")
        std_rec = np.std(w_rec)
        print(f"\nStandard deviation for {pop} recurrent weights: {std_rec:.4f}\n")
        std_rec_norm = np.std(w_rec) / np.mean(w_rec)
        print(f"\nStandard deviation normalized for {pop} recurrent weights: {std_rec_norm:.4f}\n")
        skew_rec = skew(w_rec)
        print(f"\nSkewness for {pop} recurrent weights: {skew_rec:.4f}\n")
        kurt_rec = kurtosis(w_rec)
        print(f"\nKurtosis for {pop} recurrent weights: {kurt_rec:.4f}\n")

        if w_rec.size > 0:
            pooled_list.append(w_rec)
        if w_inc.size > 0:
            pooled_list.append(w_inc)

        # stessa cosa per i pesi iniziali, se ci sono
        if initial_weights_dict is not None and pop in initial_weights_dict:
            w_rec0 = _clean(initial_weights_dict[pop]['recurrent'])
            w_inc0 = _clean(initial_weights_dict[pop]['incoming'])
            cleaned_init[pop] = {'recurrent': w_rec0, 'incoming': w_inc0}

            # i bin devono coprire anche i pesi iniziali
            if w_rec0.size > 0:
                pooled_list.append(w_rec0)
            if w_inc0.size > 0:
                pooled_list.append(w_inc0)

            # controllo: iniziali e finali devono essere le STESSE sinapsi
            if w_rec0.size != w_rec.size or w_inc0.size != w_inc.size:
                print("  ATTENZIONE ({}): numero di sinapsi diverso fra t=0 e "
                      "t finale (ricorrenti {} vs {}, in ingresso {} vs {}). "
                      "Gli istogrammi non sono direttamente confrontabili in "
                      "conteggi.".format(pop, w_rec0.size, w_rec.size,
                                         w_inc0.size, w_inc.size))

    if not pooled_list:
        print("No valid data to plot after cleaning.")
        return

    pooled = np.concatenate(pooled_list)

    # --- 2. Asse X comune e bin edges comuni ---
    if xlim is not None:
        lo, hi = xlim
    else:
        lo = pooled.min()
        hi = pooled.max()
        margin = 0.02 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - margin, hi + margin

    edges = np.linspace(lo, hi, nbins + 1)

    # --- 3. Setup della griglia (Righe = Popolazioni, Colonne = 2) ---
    cols = 2
    rows = num_pops

    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))

    # Assicura che axes sia indirizzabile in 2D anche se c'e' una sola riga
    if rows == 1:
        axes = np.atleast_2d(axes)

    ymax_final = 0.0
    ymax_all = 0.0

    # --- 4. Disegno dei Plot ---
    for i, (pop, data) in enumerate(cleaned.items()):
        ax_rec = axes[i, 0]
        ax_inc = axes[i, 1]

        init = cleaned_init.get(pop, None)

        def plot_panel(ax, weights, weights_init, title_suffix):
            nonlocal ymax_final, ymax_all

            # prima i pesi iniziali, in grigio e sotto
            if weights_init is not None and weights_init.size > 0:
                c0, _, _ = ax.hist(weights_init, bins=edges, density=density,
                                   color='0.65', alpha=0.55, edgecolor='0.4',
                                   linewidth=0.5, zorder=1, label=label_initial)
                ymax_all = max(ymax_all, c0.max())

            # poi i pesi finali, sopra
            if weights.size > 0:
                counts, _, _ = ax.hist(weights, bins=edges, density=density,
                                       alpha=0.7, color='steelblue',
                                       edgecolor='black', linewidth=0.5,
                                       zorder=2, label=label_final)
                ymax_final = max(ymax_final, counts.max())
                ymax_all = max(ymax_all, counts.max())
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=labelsize, color='gray')

            ax.set_xlim(lo, hi)
            ax.set_title(f"{pop} - {title_suffix}", fontsize=titlesize)
            # numeri ed etichetta sull'asse x su TUTTI i pannelli
            ax.tick_params(axis='both', labelsize=labelsize, labelbottom=True)
            ax.set_xlabel("Synaptic weight (pA)", fontsize=labelsize)
            if logy:
                ax.set_yscale('log')

            # legenda in ogni pannello, ma solo se c'e' qualcosa da etichettare
            # (un pannello senza sinapsi non ha handle e matplotlib avviserebbe)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right', fontsize=labelsize)

        plot_panel(ax_rec, data['recurrent'],
                   init['recurrent'] if init else None, "Ricorrenti")
        plot_panel(ax_inc, data['incoming'],
                   init['incoming'] if init else None, "Da altre pop.")

        # Etichetta asse Y (solo nella colonna di sinistra)
        ax_rec.set_ylabel("Density" if density else "Count", fontsize=labelsize)

    # --- 5. Limiti asse Y ---
    if scale_y_on_final and ymax_final > 0 and not logy:
        # il picco dei pesi iniziali esce dal grafico, la distribuzione
        # finale resta leggibile
        for ax in axes.flatten():
            ax.set_ylim(0, ymax_final * 1.05)
    elif sharey:
        top = ymax_final if scale_y_on_final else ymax_all
        for ax in axes.flatten():
            ax.set_ylim(0, top * 1.05)

    plt.tight_layout()

    # Salvataggio
    plt.savefig(data_path + "weights_histogram_grouped" + num + ".png")
    plt.draw()

overlap = network_params["overlap"]
srs = load_spike_data(overlap = overlap)

sr0 = srs[0]
sr1 = srs[1]
sr2 = srs[2]
sr3 = srs[3]
sr4 = srs[4]
sr5 = srs[5] if len(srs) > 5 else None
sr6 = srs[6] if len(srs) > 6 else None

weight_dict_1 = load_synaptic_weights(data_path + "weights_"+ str(simulation_params["t_sim"]) + ".dat")
weight_dict_0 = load_synaptic_weights(data_path + "weights_0.dat")

plot_weights_histogram(weight_dict_1, data_path, num="_1")

plot_weights_histogram_combined(weight_dict_1, data_path, num="_1")

# pesi finali e pesi iniziali (weights_0.dat), raggruppati per popolazione  
grouped_weights = group_weights(weight_dict_1)
grouped_weights_0 = group_weights(weight_dict_0)

plot_weights_histogram_grouped(grouped_weights, data_path, num="_1", nbins=50,
                               initial_weights_dict=grouped_weights_0,
                               xlim=None, sharey=False, density=False,
                               scale_y_on_final=False, logy=False)


# print("data path: ", data_path)

# raster_plot(data_path)

# plt.show()

start_time_before = 0
stop_time_before = 6000.0

plot_instantaneus_firing_rate(srs)

# expected population sizes, so that neurons which never fire are still counted
n_sel = int(network_params["N_exc"] * network_params["f"])
n_nonsel = network_params["N_exc"] - network_params["p"] * n_sel
n_inh = network_params["N_inh"]

# firing_rate() returns (ids, rates): the histogram only needs the rates,
# hence the [1].
firing_rates_dict_before = {
    "Selective population 0": firing_rate(sr=sr0, t_start=start_time_before, t_stop=stop_time_before, n_neurons=n_sel)[1],
    "Selective population 1": firing_rate(sr=sr1, t_start=start_time_before, t_stop=stop_time_before, n_neurons=n_sel)[1],
    "Selective population 2": firing_rate(sr=sr2, t_start=start_time_before, t_stop=stop_time_before, n_neurons=n_sel)[1],
    "Selective population 3": firing_rate(sr=sr3, t_start=start_time_before, t_stop=stop_time_before, n_neurons=n_sel)[1],
    "Selective population 4": firing_rate(sr=sr4, t_start=start_time_before, t_stop=stop_time_before, n_neurons=n_sel)[1],
    "Non selective population": firing_rate(sr=sr5, t_start=start_time_before, t_stop=stop_time_before, n_neurons=n_nonsel)[1] if sr5 is not None else [],
    "Inhibitory population": firing_rate(sr=sr6, t_start=start_time_before, t_stop=stop_time_before, n_neurons=n_inh)[1] if sr6 is not None else []
}

plot_firing_rate_histogram(firing_rates_dict_before, data_path, filename="firing_rate_separated_before",
                           duration_ms=stop_time_before - start_time_before)


start_time_after = 9000.0
stop_time_after = simulation_params["t_sim"]

firing_rates_dict_after = {
    "Selective population 0": firing_rate(sr=sr0, t_start=start_time_after, t_stop=stop_time_after)[1],
    "Selective population 1": firing_rate(sr=sr1, t_start=start_time_after, t_stop=stop_time_after)[1],
    "Selective population 2": firing_rate(sr=sr2, t_start=start_time_after, t_stop=stop_time_after)[1],
    "Selective population 3": firing_rate(sr=sr3, t_start=start_time_after, t_stop=stop_time_after)[1],
    "Selective population 4": firing_rate(sr=sr4, t_start=start_time_after, t_stop=stop_time_after)[1],
    "Non selective population": firing_rate(sr=sr5, t_start=start_time_after, t_stop=stop_time_after)[1] if sr5 is not None else [],
    "Inhibitory population": firing_rate(sr=sr6, t_start=start_time_after, t_stop=stop_time_after, n_neurons=n_inh)[1] if sr6 is not None else []
}

plot_firing_rate_histogram(firing_rates_dict_after, data_path, filename="firing_rate_separated_after",
                           duration_ms=stop_time_after - start_time_after)


sr_excitatory = np.vstack([sr0, sr1, sr2, sr3, sr4, sr5]) if sr5 is not None else np.vstack([sr0, sr1, sr2, sr3, sr4])

firing_rates_dict_before_combined = {
    "Excitatory populations": firing_rate(sr=sr_excitatory, t_start=start_time_before, t_stop=stop_time_before, n_neurons=n_sel*5+n_nonsel)[1],
    "Inhibitory population": firing_rate(sr=sr6, t_start=start_time_before, t_stop=stop_time_before, n_neurons=n_inh)[1] if sr6 is not None else []
}

plot_firing_rate_histogram(firing_rates_dict_before_combined, data_path, filename="firing_rate_combined_before",
                           duration_ms=stop_time_before - start_time_before)


firing_rates_dict_after_combined = {
    "Excitatory populations": firing_rate(sr=sr_excitatory, t_start=start_time_after, t_stop=stop_time_after, n_neurons=n_sel*5+n_nonsel)[1],
    "Inhibitory population": firing_rate(sr=sr6, t_start=start_time_after, t_stop=stop_time_after, n_neurons=n_inh)[1] if sr6 is not None else []
}

plot_firing_rate_histogram(firing_rates_dict_after_combined, data_path, filename="firing_rate_combined_after",
                           duration_ms=stop_time_after - start_time_after)


# plt.show()