#!/bin/bash

# Crea la cartella di destinazione finale se non esiste
mkdir -p data_seed_fromstdp


for i in {0..9}; do
    seed=$((143202461 + i))
    echo "========================================"
    echo "Avvio simulazione: Seed = $seed"
    echo "========================================"

    # Esegui il modello Python
    python3 run_model.py --seed "$seed"

     if [ -d "data/" ]; then
        # Rinomina e sposta la cartella come richiesto
        mv data/ "data_seed_fromstdp/data_${seed}"
        echo "-> Salvato in: data_seed_fromstdp/data_${seed}"
    else
        echo "-> ATTENZIONE: La cartella data/ non è stata creata per seed=$seed."
        # exit 1  # Decommenta questa riga se vuoi che lo script si fermi in caso di errore
    fi
done

echo "Finito il ciclo di simulazioni. Tutti i dati sono stati salvati in data_seed_fromstdp/."