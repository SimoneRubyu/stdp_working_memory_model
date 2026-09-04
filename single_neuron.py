import nest
import numpy as np
import matplotlib.pyplot as plt

# Ripristina il kernel
nest.ResetKernel()

# ==========================================
# 1. DIZIONARI DEI PARAMETRI (Pannello di controllo)
# ==========================================

# Parametri topologici della rete
network_params = {
    "N_parrots": 2000,
    "N_exc": 1600,
    "N_inh": 400
}

# Impostazioni MIP basate su Processi Figlio e Correlazione
user_mip_settings = {
    "child_rate": 950.0,     # Frequenza desiderata per i singoli parrot neuron (Hz)
    "correlation": 0.1     # Correlazione tra i treni di spike (valore tra >0 e 1)
}

# Traduzione in parametri nativi per NEST
assert user_mip_settings["correlation"] > 0, "La correlazione deve essere maggiore di 0 per usare il MIP generator."
mip_params = {
    "rate": user_mip_settings["child_rate"] / user_mip_settings["correlation"],
    "p_copy": user_mip_settings["correlation"]
}

# Parametri del neurone target LIF (iaf_psc_exp)
lif_params = {
    "V_m": 0.0,        
    "E_L": 0.0,        
    "tau_m": 15.0,       
    "t_ref": 2.0,        
    "V_th": 20.0,       
    "V_reset": 0.0,    
    "tau_minus": 20.0    # <-- Costante di tempo per la traccia post-sinaptica (LTD)
}

# Parametri per le sinapsi eccitatorie (STDP)
stdp_params = {
    "synapse_model": "stdp_synapse",
    "weight": 5.0,       
    "delay": 0.55,        
    "Wmax": 100.0,       
    "alpha": 1.1,        
    "lambda": 0.01,      
    "mu_plus": 0.4,      
    "mu_minus": 1.0,     
    "tau_plus": 20.0     # <-- Costante di tempo per la traccia pre-sinaptica (LTP)
}

# Parametri per le sinapsi inibitorie (Statiche)
static_params = {
    "synapse_model": "static_synapse",
    "weight": -20.0,     
    "delay": 0.55
}


# ==========================================
# 2. CREAZIONE DEI NODI
# ==========================================
# Creazione del MIP generator
mip = nest.Create("mip_generator", params=mip_params)

# Creazione dei parrot neurons e divisione tramite slicing
parrots = nest.Create("parrot_neuron", network_params["N_parrots"])
parrots_exc = parrots[:network_params["N_exc"]]
parrots_inh = parrots[network_params["N_exc"]:]

# Creazione del neurone target LIF
target_neuron = nest.Create("iaf_psc_exp", 1, params=lif_params)

# Strumenti di misurazione
spike_recorder_target = nest.Create("spike_recorder")
sr_exc = nest.Create("spike_recorder")  # Nuovo: registra i parrot eccitatori
sr_inh = nest.Create("spike_recorder")  # Nuovo: registra i parrot inibitori


# ==========================================
# 3. CONNESSIONI
# ==========================================
# Connessione MIP -> Parrots (tutti)
nest.Connect(mip, parrots)

# Connessione Parrots -> Target 
nest.Connect(parrots_exc, target_neuron, syn_spec=stdp_params)
nest.Connect(parrots_inh, target_neuron, syn_spec=static_params)

# Selezioniamo solo il 50% dei parrot neuron usando lo slicing [::2] 
# (prende un neurone ogni due, mantenendo la distribuzione spaziale degli ID)
parrots_exc_to_record = parrots_exc[::2]
parrots_inh_to_record = parrots_inh[::2]

# Connessione dei registratori: sr_exc e sr_inh ascoltano solo il 50%
nest.Connect(target_neuron, spike_recorder_target)
nest.Connect(parrots_exc_to_record, sr_exc)
nest.Connect(parrots_inh_to_record, sr_inh)


# ==========================================
# 4. ESECUZIONE SIMULAZIONE
# ==========================================
sim_time = 1000.0 # ms
nest.Simulate(sim_time)


# ==========================================
# 5. ESTRAZIONE DATI E RASTER PLOT
# ==========================================
import matplotlib.pyplot as plt

# Estrazione eventi
events_target = nest.GetStatus(spike_recorder_target, 'events')[0]
events_exc = nest.GetStatus(sr_exc, 'events')[0]
events_inh = nest.GetStatus(sr_inh, 'events')[0]

plt.figure(figsize=(12, 7))

# Plot dei parrot eccitatori (Rossi, 50% della popolazione)
plt.scatter(events_exc['times'], events_exc['senders'], 
            color='red', s=2, marker='.', label='Parrots Eccitatori (50%)')

# Plot dei parrot inibitori (Neri, 50% della popolazione)
plt.scatter(events_inh['times'], events_inh['senders'], 
            color='black', s=2, marker='.', label='Parrots Inibitori (50%)')

# Plot del target LIF (Trattino blu ben visibile)
plt.scatter(events_target['times'], events_target['senders'], 
            color='blue', s=300, marker='|', linewidths=2.5, zorder=3, label='Target LIF')

# Formattazione del grafico
plt.title('Raster Plot (50% dei neuroni input registrati)', fontsize=14)
plt.xlabel('Tempo (ms)', fontsize=12)
plt.ylabel('ID Neurone', fontsize=12)

# Estraiamo l'ID minimo e massimo in modo sicuro per i margini dell'asse Y
y_min = min(parrots.tolist()) - 10
y_max = target_neuron.tolist()[0] + 10
plt.ylim(y_min, y_max)

plt.xlim(0, sim_time)
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()