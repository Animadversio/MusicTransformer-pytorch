import matplotlib.pylab as plt
import numpy as np
import music21 as M2
# cond_file = r"E:\Datasets\classical-music-midi\mozart\mz_332_2.mid"
# mid = pretty_midi.PrettyMIDI(midi_file=cond_file)
# ps_roll = mid.get_piano_roll()
fs = 1 / 100.0
def plot_piano_roll(ps_roll, fs = 1/100.0):
    ps_uniq = ps_roll.nonzero()[0]
    minps = ps_uniq.min()
    maxps = ps_uniq.max()
    minps = np.int(np.floor(minps / 12)) * 12
    maxps = np.int(np.ceil(maxps / 12)) * 12
    maxT = ps_roll.shape[1] * fs
    # octv_ticks = list(range(int(minps), int(maxps), 12))
    octv_ticks = list(range(int(0), int(120), 12))
    T_ticks = list(range(0, int(maxT), 10))
    figh = plt.figure(figsize=[0.15*maxT, 7 / 128 * (maxps - minps)])
    plt.imshow(ps_roll[:, :], cmap='gray', aspect='auto')
    plt.hlines(octv_ticks, plt.xlim()[0], plt.xlim()[1], alpha=0.30, colors='white')
    plt.gca().invert_yaxis()
    plt.yticks(octv_ticks, [M2.pitch.Pitch(p).nameWithOctave for p in octv_ticks])
    plt.xticks([t / fs for t in T_ticks], T_ticks)
    figh.gca().set_ylim(minps, maxps)
    #figh.show()
    return figh