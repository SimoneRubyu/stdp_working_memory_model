#!/bin/bash

# Crea la cartella di destinazione finale se non esiste
mkdir -p datas_mu

# LC_ALL=C forza seq a usare il punto decimale invece della virgola
for mu in $(LC_ALL=C seq 0.0 0.1 1.0); do
    echo "========================================"
    echo "Avvio simulazione: mu = $mu"
    echo "========================================"
    python run_model.py --mu "$mu"
    python figure2_stdp.py --path data/ > data/output_figure2.txt
    python analysis2_stdp.py > data/output_analysis2.txt

    if [ -d "data/" ]; then
        # Rinomina e sposta la cartella come richiesto
        mv data/ "datas_mu/data_${mu}"
        echo "-> Salvato in: datas_mu/data_${mu}"
    else
        echo "-> ATTENZIONE: La cartella data/ non è stata creata per mu=$mu."
        # exit 1  # Decommenta questa riga se vuoi che lo script si fermi in caso di errore
    fi  
done

echo "simulazioni completate!"