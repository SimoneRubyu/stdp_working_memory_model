#!/bin/bash

# Crea la cartella di destinazione finale se non esiste
mkdir -p datas_init_std

# LC_ALL=C forza seq a usare il punto decimale invece della virgola
for init_std in $(LC_ALL=C seq 0.00225 0.001 0.01225); do
    echo "========================================"
    echo "Avvio simulazione: init_std = $init_std"
    echo "========================================"
    python run_model.py --init_std "$init_std"
    python figure2_stdp.py --path data/ > data/output_figure2.txt
    python analysis2_stdp.py > data/output_analysis2.txt

    if [ -d "data/" ]; then
        # Rinomina e sposta la cartella come richiesto
        mv data/ "datas_init_std/data_${init_std}"
        echo "-> Salvato in: datas_init_std/data_${init_std}"
    else
        echo "-> ATTENZIONE: La cartella data/ non è stata creata per init_std=$init_std."
        # exit 1  # Decommenta questa riga se vuoi che lo script si fermi in caso di errore
    fi  
done

echo "simulazioni completate!"