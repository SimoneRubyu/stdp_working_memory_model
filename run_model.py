from model.model_stdp import STDPModel
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from model.model_helpers import get_weight, noise_params, get_rate_and_weight_poisson, lognormal_params

# Get and check path and rng seed
parser = ArgumentParser()
parser.add_argument("--path", type=str, default=None, help='Path for the simulation output (default: data).')
parser.add_argument("--seed", type=int, default=143202461, help='Seed for random number generation (default: 143202461).')
parser.add_argument("--background", type=float, default=19.7, help='Average variation of membrane potential elicited by external current (default: 18.5 mV).')
parser.add_argument("--cue", type=float, default=1.15, help='Contrast factor for the cue stimulus (default: 1.15).')
parser.add_argument("--time_cue", type=float, default=350.0, help='Duration of the cue stimulus in ms (default: 350.0 ms).')
parser.add_argument("--learning_rate", type=float, default=0.1, help='Learning rate for STDP (default: 0.1).')
parser.add_argument("--asymmetry", type=float, default=1.0, help='Asymmetry factor for STDP depression and potentiation (default: 1.0).')
parser.add_argument("--mu", type=float, default=1.0, help='Exponent for weight dependence of STDP (default: 1.0).')
args = parser.parse_args()

if args.path is None:
    data_path = os.path.join(os.getcwd(), 'data/')
else:
    data_path = os.path.join(os.getcwd(), args.path+"/")

# Script needed to run the model.
#         eta_ext
# Fig 2A - 22.7 (single stable state activity)
# Fig 2B - 23.7 (bi-stable activity with synchronous spiking activity)
# Fig 2C - 24.1 (bi-stable activity with asynchronous spiking activity)

# average variation of membrane potential elicited by external current [mV]
eta_exc = 19.7
# network params dict
# here add the parameters to be edited. The rest of the parameters are in model/default_params.py
network_p = {
    # network parameters
    'N_exc': 800,
    'N_inh': 200,
    # excitatory input current [mV]
    'eta_exc': eta_exc,
    'poisson_bkg':{'allow': True},
    # current used to go back to the spontaneous activity
    'eta_exc_end': eta_exc - eta_exc,
    # synaptic parameters
    'syn_params' : {'autapses' : True, 'multapses' : True, 
                    "J_b" : 0.10, "J_p" : 0.10,
                    "start_dist_weights" : {"allow" : True, "std": 0.01}},
    # STDP parameters
    'stdp_params' : {'tau_plus' : 20.0, 'tau_minus' : 20.0, 
                     'lambda' : 0.04, 'alpha' : 1.0,
                     'mu_plus' : 1.0, 'mu_minus' : 1.0,
                     'Wmax' : 150.0},
    # item loading parameters
    'stimulation_params' : {'T_cue' : 3000.0, 'A_cue' : 1.15 , "correlation_c": 0.45}}

# presimulation time (i.e. time in which the network stays in the spontaneous activity)
tpresim = 3000.0
# simulation time
tsim = 6000.0
# time to stop stdp learning (in ms, if None stdp is active during the whole simulation)
t_stop_stdp = 4500.0

# simulation params dict
# here add the parameters to be edited. The rest of the parameters are in default_params.py
simulation_p = {
    # path to data
    "data_path" : data_path,
    # master seed
    #"master_seed" : 143202463,
    "master_seed" : args.seed,
    # number of OpenMP threads
    "threads" : 8,
    # overall simulation time
    "t_sim" : tsim + tpresim,
    # beginning of th current stimulus which diminishes overall background input
    "eta_end_origin": tsim + tpresim - 800.0,
    "recording_params" : {
        # fraction of neurons recorded for each selective population
        "fraction_pop_recorded" : 1.0,
        # fraction of neurons recorded for weight distribution
        "fraction_weights_recorded" : 0.01,
        # selective excitatory population recorded (0, ..., p-1)
        "pop_recorded" : [0, 1, 2, 3, 4, 5],
        "spike_recording_params": {"start": 100.0},
        # save spike data to file
        "save_to_file" : True,
        # save final synaptic weights to file
        "save_weights" : True,
        # save mip_generator spike times
        "save_mip_spikes" : False
    },
    # time to stop stdp learning (in ms, if None stdp is active during the whole simulation)
    "t_stop_stdp" : t_stop_stdp
}

# initialize the network model
network = STDPModel(network_p, simulation_p)

# add background
network.add_background_input(start=0.0, stop=tsim+tpresim)

# to reproduce Figure 1A
#network.add_item_loading_signals(pop_id=[0], origin=[tpresim])
#network.add_nonspecific_readout_signal(origin=[tpresim+1100.0])

# to reproduce Figure 1B and 1C
network.add_item_loading_signals(pop_id=[0], origin=[tpresim])

# add item loading signals with mip generator
# network.add_item_loading_signals_mip(pop_id=[0], origin=[tpresim])

# save used parameters into a json
network.save_params()

# build network
network.build_network()

# save initial synaptic weights to file
# network.save_weights(time=0.0)

# simulate network
network.simulate_network()

# save data to file
network.save_spike_data()

# save final synaptic weights to file
network.save_weights(time=tsim+tpresim)

# plots a raster plot of all the neurons recorded
network.raster_plot()

# plt.show()
