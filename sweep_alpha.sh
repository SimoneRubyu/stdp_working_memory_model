#!/bin/bash

# Crea la cartella di destinazione finale se non esiste
mkdir -p datas_alpha

# LC_ALL=C forza seq a usare il punto decimale invece della virgola
for alpha in $(LC_ALL=C seq 0.5 0.1 1.5); do
    echo "========================================"
    echo "Avvio simulazione: alpha = $alpha"
    echo "========================================"
    python run_model.py --asymmetry "$alpha"
    python figure2_stdp.py --path data/ > data/output_figure2.txt
    python analysis2_stdp.py > data/output_analysis2.txt

    if [ -d "data/" ]; then
        # Rinomina e sposta la cartella come richiesto
        mv data/ "datas_alpha/data_${alpha}"
        echo "-> Salvato in: datas_alpha/data_${alpha}"
    else
        echo "-> ATTENZIONE: La cartella data/ non è stata creata per alpha=$alpha."
        # exit 1  # Decommenta questa riga se vuoi che lo script si fermi in caso di errore
    fi  
done

echo "simulazioni completate!"