#!/bin/bash

for cartella in datas_init_std/data_*/; do
    # Rimuove il percorso per stampare solo il nome della cartella
    nome_cartella=datas_init_std/$(basename "$cartella")
    echo "Trovata cartella: $nome_cartella"
    python analysis2_stdp.py --path "$nome_cartella" > "$nome_cartella/output_analysis2.txt"
done