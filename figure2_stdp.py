#!/usr/bin/env python3
"""
figure2_stdp.py
===============

Figura "stile fig. 2 di Mongillo" per le simulazioni con STDP.

Tre pannelli:
  (a) firing rate istantaneo della popolazione targettata (default: selective
      pop 0) e di quella non targettata (default: selective pop 1, oppure
      tutte le altre eccitatorie combinate, esclusa l'inibitoria);
  (b) raster plot dell'intera simulazione: spontanea pre-stimolo, stimolo,
      spontanea post-stimolo (senza le curve x e u della STP);
  (c) istogramma della differenza di firing rate per neurone fra la finestra
      DOPO lo stimolo e quella PRIMA dello stimolo, nella popolazione
      targettata.

Uso:
    python figure2_stdp.py --path data/
    python figure2_stdp.py --path data/ --nontarget all_exc --binwidth 50

Le funzioni sono scritte per essere anche importabili: se preferisci, puoi
copiare `inst_rate` e `figure2` dentro analysis2_stdp.py, dove `srs`,
`network_params`, `simulation_params` e `firing_rate` esistono gia'.
"""

import os
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Caricamento dati (stessa logica di analysis2_stdp.py)
# ---------------------------------------------------------------------------

def load_params(data_path):
    with open(os.path.join(data_path, 'network_params.json'), 'r') as f:
        network_params = json.load(f)
    with open(os.path.join(data_path, 'simulation_params.json'), 'r') as f:
        simulation_params = json.load(f)
    return network_params, simulation_params


def load_spike_data(data_path, network_params, overlap=False):
    """Ritorna la lista srs: p popolazioni selettive, poi non selettiva e
    inibitoria se i relativi file esistono."""
    n_spike_dats = network_params["p"]
    if os.path.isfile(os.path.join(data_path, "spikedata5.dat")):
        n_spike_dats += 1
    if os.path.isfile(os.path.join(data_path, "spikedata6.dat")):
        n_spike_dats += 1

    if not overlap:
        srs = [np.loadtxt(os.path.join(data_path, "spikedata%d.dat" % i))
               for i in range(n_spike_dats)]
    else:
        # rinumera gli id in modo che la pop selettiva i abbia id [800*i, 800*(i+1))
        old_data = [np.loadtxt(os.path.join(data_path, "spikedata%d.dat" % i))
                    for i in range(network_params["p"])]
        ids = np.loadtxt(os.path.join(data_path, "selective_pop_ids.dat"))
        srs = []
        for i in range(len(old_data)):
            sorted_ids = np.sort(ids[:, i]) + 1
            sr = old_data[i]
            for idx in sorted_ids:
                pos_old_ids = np.where(old_data[i][:, 0] == idx)
                if list(pos_old_ids[0]) != []:
                    for r in list(pos_old_ids[0]):
                        sr[r, 0] = np.where(sorted_ids == idx)[0][0] + 800 * i
            srs.append(sr)
        # le popolazioni non selettiva/inibitoria non vengono rinumerate
        for i in range(network_params["p"], n_spike_dats):
            srs.append(np.loadtxt(os.path.join(data_path, "spikedata%d.dat" % i)))

    return srs


# ---------------------------------------------------------------------------
# Firing rate per neurone (identica a quella di analysis2_stdp.py)
# ---------------------------------------------------------------------------

def firing_rate(t_start, t_stop, sr, id_min=None, n_neurons=None, verbose=True):
    """Firing rate (Hz) di ogni neurone di una popolazione nella finestra
    [t_start, t_stop). I neuroni silenti restano a 0 e NON vengono scartati."""
    duration_s = (t_stop - t_start) / 1000.0
    if duration_s <= 0:
        raise ValueError("t_stop must be strictly greater than t_start")

    if sr is None or np.size(sr) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=float)

    sr = np.atleast_2d(np.asarray(sr, dtype=float))
    all_senders = np.rint(sr[:, 0]).astype(np.int64)
    times = sr[:, 1]

    if id_min is None:
        id_min = int(all_senders.min())
    if n_neurons is None:
        n_neurons = int(all_senders.max() - id_min + 1)
    ids = np.arange(id_min, id_min + n_neurons, dtype=np.int64)

    mask = (times >= t_start) & (times < t_stop)
    senders = all_senders[mask]

    counts = np.zeros(n_neurons, dtype=np.int64)
    if senders.size:
        idx = senders - id_min
        keep = (idx >= 0) & (idx < n_neurons)
        if not keep.all():
            print("  WARNING: {} spikes fall outside the id range [{}, {}] "
                  "and were discarded.".format(int((~keep).sum()), id_min,
                                               id_min + n_neurons - 1))
        counts += np.bincount(idx[keep], minlength=n_neurons)

    rates = counts / duration_s

    if verbose:
        print("Firing rate on [{}, {}) ms | {} neurons | mean {:.2f} Hz | "
              "median {:.2f} Hz | max {:.2f} Hz | silent {}".format(
                  t_start, t_stop, n_neurons, rates.mean(), np.median(rates),
                  rates.max(), int((rates == 0).sum())))

    return ids, rates


# ---------------------------------------------------------------------------
# Firing rate istantaneo (versione compatta di instantaneus_firing_rate)
# ---------------------------------------------------------------------------

def inst_rate(sr, n_neurons, binwidth=25.0, t_min=0.0, t_max=None):
    """Firing rate istantaneo di popolazione: spike per bin / (bin in s) / n_neurons.

    Ritorna (centri dei bin [ms], rate [Hz]). Bin comuni a tutte le popolazioni,
    cosi' le curve sono confrontabili una a una.
    """
    if sr is None or np.size(sr) == 0:
        return np.array([]), np.array([])

    sr = np.atleast_2d(np.asarray(sr, dtype=float))
    times = sr[:, 1]

    if t_max is None:
        t_max = float(times.max())

    edges = np.arange(t_min, t_max + binwidth, binwidth)
    counts, _ = np.histogram(times, bins=edges)
    rate = counts / (binwidth / 1000.0) / float(n_neurons)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, rate


# ---------------------------------------------------------------------------
# Finestra dello stimolo
# ---------------------------------------------------------------------------

def get_stim_window(network_params, simulation_params, fallback=(6000.0, 9000.0)):
    """Ricava (inizio, fine) dello stimolo da network_params, con fallback."""
    t_sim = float(simulation_params.get("t_sim", np.inf))

    for key in ("item_loading", "seq_item_loading"):
        if key in network_params:
            block = network_params[key]
            try:
                t0 = float(block["origin"][0])
            except (KeyError, IndexError, TypeError):
                continue

            t1 = None
            if "stop" in block:
                stop = float(np.atleast_1d(block["stop"])[0])
                # in analysis2_stdp.py 'stop' e' usata come DURATA (origin + stop);
                # se cosi' si sfora t_sim la interpreto come istante assoluto
                t1 = t0 + stop if (t0 + stop) <= t_sim else stop
            elif "T_cue" in network_params.get("stimulation_params", {}):
                t1 = t0 + float(network_params["stimulation_params"]["T_cue"])

            if t1 is not None and t1 > t0:
                return t0, t1

    print("  ATTENZIONE: finestra dello stimolo non trovata nei parametri, "
          "uso il default {}.".format(fallback))
    return fallback


# ---------------------------------------------------------------------------
# La figura
# ---------------------------------------------------------------------------

def figure2(srs, network_params, simulation_params, data_path=".",
            target_pop=0, nontarget="pop1", binwidth=25.0,
            t_stim=None, t_before=None, t_after=None,
            hist_bins=40, hist_range=None,
            raster_frac=1.0, raster_nontarget_pop=None, markersize=1.2,
            filename="fig2_stdp", save_pdf=True, dpi=300,
            labelsize=22, panel_labels=("(a)", "(b)", "(c)")):
    """Costruisce la figura a tre pannelli.

    Parametri principali
    --------------------
    target_pop : int
        Indice della popolazione selettiva stimolata (la tua e' la 0).
    nontarget : "pop1" | int | "all_exc"
        Popolazione di confronto. "all_exc" = tutte le eccitatorie tranne la
        targettata (selettive rimanenti + non selettiva), esclusa l'inibitoria.
    binwidth : float
        Larghezza del bin del firing rate istantaneo, in ms.
    t_stim, t_before, t_after : tuple (t0, t1) o None
        Finestre in ms. Se None vengono dedotte: stimolo da network_params,
        'before' = i 6000 ms che precedono lo stimolo, 'after' = dalla fine
        dello stimolo a t_sim.
    hist_bins : int
        Numero MASSIMO di bin in (c). I bordi vengono allineati al quanto
        1000/durata_finestra, altrimenti l'istogramma esce "a pettine".
    hist_range : tuple o None
        Range dell'istogramma di (c). Se None viene scelto sui dati.
    raster_frac : float
        Frazione di neuroni per popolazione da disegnare nel raster (1.0 =
        tutti). Con 15 s di simulazione, 0.2-0.3 rende il pannello piu' leggibile.
    raster_nontarget_pop : int o None
        Popolazione non targettata da mostrare nel raster. Se None usa quella
        del confronto in (a), oppure la prima disponibile se nontarget="all_exc".
    """
    # ---------------- dimensioni delle popolazioni -------------------------
    N_exc = network_params["N_exc"]
    f = network_params["f"]
    p = network_params["p"]
    n_sel = int(N_exc * f)
    n_nonsel = N_exc - p * n_sel
    t_sim = float(simulation_params["t_sim"])

    has_nonsel = len(srs) > p
    sr_target = srs[target_pop]

    # ---------------- finestre temporali -----------------------------------
    if t_stim is None:
        t_stim = get_stim_window(network_params, simulation_params)
    t_stim_start, t_stim_stop = float(t_stim[0]), float(t_stim[1])

    if t_before is None:
        t_before = (max(0.0, t_stim_start - 6000.0), t_stim_start)
    if t_after is None:
        t_after = (t_stim_stop, min(t_stim_stop + 6000.0, t_sim))
    t_before = (float(t_before[0]), float(t_before[1]))
    t_after = (float(t_after[0]), float(t_after[1]))

    print("\nFinestre usate:")
    print("  spontanea pre-stimolo : [{:.0f}, {:.0f}) ms".format(*t_before))
    print("  stimolo               : [{:.0f}, {:.0f}) ms".format(t_stim_start, t_stim_stop))
    print("  spontanea post-stimolo: [{:.0f}, {:.0f}) ms\n".format(*t_after))

    # ---------------- popolazione non targettata ---------------------------
    if nontarget == "all_exc":
        others = [srs[i] for i in range(p) if i != target_pop]
        n_other = n_sel * (p - 1)
        if has_nonsel and np.size(srs[p]) > 0:
            others.append(srs[p])
            n_other += n_nonsel
        others = [np.atleast_2d(np.asarray(s, dtype=float))
                  for s in others if s is not None and np.size(s) > 0]
        sr_nontarget = np.vstack(others) if others else None
        nontarget_label = "Nontargeted pops. (all exc.)"
    else:
        idx = 1 if nontarget == "pop1" else int(nontarget)
        sr_nontarget = srs[idx]
        n_other = n_sel
        nontarget_label = "Nontargeted pop. {}".format(idx)

    # ---------------- layout ------------------------------------------------
    fig, axs = plt.subplot_mosaic(
        [['rate', 'hist'], ['raster', 'hist']],
        figsize=(18, 8),
        gridspec_kw={'wspace': 0.28, 'hspace': 0.10,
                     'width_ratios': [3, 1.5], 'height_ratios': [2, 3]})

    ax_rate = axs['rate']
    ax_rast = axs['raster']
    ax_hist = axs['hist']

    col_target = "limegreen"
    col_other = "black"

    # ================= (a) firing rate istantaneo ==========================
    t_t, r_t = inst_rate(sr_target, n_sel, binwidth=binwidth, t_min=0.0, t_max=t_sim)
    t_o, r_o = inst_rate(sr_nontarget, n_other, binwidth=binwidth, t_min=0.0, t_max=t_sim)

    ax_rate.plot(t_t, r_t, color=col_target, linewidth=1.2, label="Targeted pop. {}".format(target_pop))
    ax_rate.plot(t_o, r_o, color=col_other, linewidth=1.2, label=nontarget_label)

    ax_rate.set_ylabel("rate [Hz]", color="k", fontsize=labelsize)
    ax_rate.tick_params(axis="x", labelbottom=False)
    ax_rate.tick_params(labelsize=labelsize, axis='y')
    ax_rate.set_xlim(0.0, t_sim)
    ax_rate.set_ylim(bottom=0.0)
    ax_rate.legend(fontsize=labelsize - 6, loc="upper right", framealpha=0.9)
    ax_rate.axvspan(t_stim_start, t_stim_stop, color='grey', alpha=0.25, zorder=0)

    # ================= (b) raster plot ======================================
    ax_rast.axvspan(t_stim_start, t_stim_stop, color='grey', alpha=0.35,
                    zorder=0, label="Stimulus")

    n_shown = max(1, int(round(n_sel * float(raster_frac))))

    def _rel(sr):
        """id relativi alla popolazione (partono da 0) e sottocampionamento."""
        sr = np.atleast_2d(np.asarray(sr, dtype=float))
        rel = sr[:, 0] - sr[:, 0].min()
        keep = rel < n_shown
        return sr[keep, 1], rel[keep]

    # popolazione non targettata da mostrare nel raster
    if raster_nontarget_pop is None:
        if nontarget == "all_exc":
            raster_nontarget_pop = 1 if target_pop != 1 else 0
        else:
            raster_nontarget_pop = 1 if nontarget == "pop1" else int(nontarget)
    sr_rast_other = srs[int(raster_nontarget_pop)]

    if sr_rast_other is not None and np.size(sr_rast_other) > 0:
        x_o, y_o = _rel(sr_rast_other)
        ax_rast.plot(x_o, y_o, '.', color=col_other, markersize=markersize,
                     alpha=0.55, rasterized=True,
                     label="Nontargeted pop. {}".format(raster_nontarget_pop))

    x_t, y_t = _rel(sr_target)
    ax_rast.plot(x_t, y_t, '.', color=col_target, markersize=markersize,
                 alpha=0.75, rasterized=True,
                 label="Targeted pop. {}".format(target_pop))

    ax_rast.set_ylabel("cell id", color="k", fontsize=labelsize)
    ax_rast.set_xlabel("Time [ms]", fontsize=labelsize)
    ax_rast.tick_params(labelsize=labelsize, pad=10, axis='x')
    ax_rast.tick_params(labelsize=labelsize, axis='y')
    ax_rast.set_ylim(0.0, n_shown)
    ax_rast.set_yticks([0, n_shown])
    ax_rast.set_yticklabels(['0', str(n_shown)])
    ax_rast.set_xlim(0.0, t_sim)

    # barrette che indicano le due finestre usate per l'istogramma (c)
    ax_rast.hlines(y=0.0, xmin=t_before[0], xmax=t_before[1],
                   color="skyblue", linewidth=7, clip_on=False)
    ax_rast.hlines(y=0.0, xmin=t_after[0], xmax=t_after[1],
                   color="orange", linewidth=7, clip_on=False)

    # ================= (c) istogramma della differenza di rate ==============
    # id_min e n_neurons identici nelle due finestre: senza questo, i neuroni
    # silenti in una sola delle due sfaserebbero l'asse degli id.
    id_min = int(np.rint(np.atleast_2d(sr_target)[:, 0].min()))

    _, rate_before = firing_rate(t_before[0], t_before[1], sr_target,
                                 id_min=id_min, n_neurons=n_sel)
    _, rate_after = firing_rate(t_after[0], t_after[1], sr_target,
                                id_min=id_min, n_neurons=n_sel)

    delta_fr = rate_after - rate_before

    # Bordi dei bin allineati al quanto dei rate. Se le due finestre hanno la
    # stessa durata T, ogni delta e' un multiplo di 1000/T Hz: con bin di
    # larghezza arbitraria meta' dei bin resterebbe vuota (istogramma "a
    # pettine"). Stessa logica gia' usata in plot_firing_rate_histogram.
    dur_b = t_before[1] - t_before[0]
    dur_a = t_after[1] - t_after[0]

    if abs(dur_b - dur_a) < 1e-9:
        quantum = 1000.0 / dur_b
        k = np.rint(delta_fr / quantum).astype(int)
        k_lo, k_hi = int(k.min()), int(k.max())
        if hist_range is not None:
            k_lo = int(np.floor(hist_range[0] / quantum))
            k_hi = int(np.ceil(hist_range[1] / quantum))
        step = max(1, int(np.ceil((k_hi - k_lo + 1) / float(hist_bins))))
        edges = (np.arange(k_lo - 0.5, k_hi + step, step)) * quantum
    else:
        if hist_range is None:
            lo = float(np.floor(min(delta_fr.min(), -1.0)))
            hi = float(np.ceil(max(delta_fr.max(), 1.0)))
            pad = 0.05 * (hi - lo)
            hist_range = (lo - pad, hi + pad)
        edges = np.linspace(hist_range[0], hist_range[1], hist_bins + 1)

    counts, bins = np.histogram(delta_fr, bins=edges)
    frac = counts / counts.sum() if counts.sum() else counts

    if hist_range is None:
        hist_range = (bins[0], bins[-1])

    ax_hist.bar(bins[:-1], frac, width=np.diff(bins), color="cornflowerblue",
                align="edge", linewidth=0.2, edgecolor="w")
    ax_hist.axvline(0.0, color="grey", linestyle=":", linewidth=1.2)
    ax_hist.set_xlabel("Firing rate difference [Hz]", fontsize=labelsize)
    ax_hist.set_ylabel("Fraction of cells", fontsize=labelsize)
    ax_hist.set_xlim(hist_range)
    ax_hist.tick_params(labelsize=labelsize)

    # ---------------- etichette dei pannelli --------------------------------
    for ax, lab in zip((ax_rate, ax_rast, ax_hist), panel_labels):
        ax.text(0.012, 0.96, lab, transform=ax.transAxes, va="top", ha="left",
                fontsize=labelsize + 1, zorder=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                          pad=1.5))

    # ---------------- statistiche per il testo della tesi -------------------
    print("\nDifferenza di firing rate (dopo - prima), popolazione {}:".format(target_pop))
    print("  media   : {:+.3f} Hz".format(delta_fr.mean()))
    print("  mediana : {:+.3f} Hz".format(np.median(delta_fr)))
    print("  std     : {:.3f} Hz".format(delta_fr.std(ddof=1)))
    print("  frazione di neuroni con delta > 0: {:.1%}".format((delta_fr > 0).mean()))
    print("  rate medio prima: {:.3f} Hz | dopo: {:.3f} Hz".format(
        rate_before.mean(), rate_after.mean()))

    plt.subplots_adjust(left=0.06, right=0.976, top=0.95, bottom=0.13)

    out_png = os.path.join(data_path, filename + ".png")
    plt.savefig(out_png, dpi=dpi)
    if save_pdf:
        plt.savefig(os.path.join(data_path, filename + ".pdf"))
    print("\nFigura salvata in {}".format(out_png))

    plt.draw()
    return fig, axs, delta_fr


# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default='data/',
                        help='Path to the data directory (default: data/).')
    parser.add_argument("--target", type=int, default=0,
                        help='Indice della popolazione targettata (default: 0).')
    parser.add_argument("--nontarget", type=str, default="pop1",
                        help='"pop1", un indice, oppure "all_exc" (default: pop1).')
    parser.add_argument("--binwidth", type=float, default=25.0,
                        help='Bin del firing rate istantaneo in ms (default: 25).')
    parser.add_argument("--bins", type=int, default=40,
                        help='Numero massimo di bin dell istogramma (default: 40).')
    parser.add_argument("--raster-frac", type=float, default=1.0,
                        help='Frazione di neuroni disegnati nel raster (default: 1.0).')
    parser.add_argument("--filename", type=str, default="fig2_stdp")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    data_path = os.path.join(os.getcwd(), args.path + "/")
    network_params, simulation_params = load_params(data_path)
    srs = load_spike_data(data_path, network_params,
                          overlap=network_params.get("overlap", False))

    figure2(srs, network_params, simulation_params, data_path=data_path,
            target_pop=args.target, nontarget=args.nontarget,
            binwidth=args.binwidth, hist_bins=args.bins,
            raster_frac=args.raster_frac, filename=args.filename, markersize=2.5, labelsize=22)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
