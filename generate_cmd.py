import custom
from custom import criterion
from custom.layers import *
from custom.config import config
from model import MusicTransformer
from data import Data
import utils
from midi_processor.processor import decode_midi, encode_midi

import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from os.path import join
from utils import find_files_by_extensions
midi_lib = [r"E:\Datasets\ecomp-midi", r"E:\Datasets\classical-music-midi", r"E:\Datasets\maestro-v2.0.0-midi\maestro-v2.0.0"]
#%%
import pretty_midi
import matplotlib.pylab as plt
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
#%%
parser = argparse.ArgumentParser(None)
parser.add_argument("-m", "--model_dir", type=str, required=True,
        help="The directory for a trained model is saved.")
parser.add_argument("-k", "--ckpt", type=str, dest="ckpt", default="final.pth",
        help="name of checkpoint file")
parser.add_argument("-i", "--cond", type=str, dest="condition_file", default=None,
        help="name of condition")
parser.add_argument("-l", "--cond_len", type=int, dest="condition_len", default=50,
        help="length of condition notes from the file")
r"E:\Datasets\ecomp-midi\ADIG01.mid"
parser.add_argument("-c", "--conf", dest="configs", default=["generate.yml"], nargs="*",
        help="A list of configuration items. "
             "An item is a file path or a 'key=value' formatted string. "
             "The type of a value is determined by applying int(), float(), and str() "
             "to it sequencially.")
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

condition_file = None
if args.condition_file is "" or args.condition_file is None:
    condition_file = None
else:
    if os.path.exists(args.condition_file):
        condition_file = args.condition_file
    else:
        print("Partial searching for ", args.condition_file)
        for midi_path in midi_lib:
            for full_fn in find_files_by_extensions(midi_path, ['.mid', '.midi']):
                if args.condition_file in full_fn:
                    condition_file = full_fn
    condition_fn = condition_file.split('\\')[-1].split('.')[0]
# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = SummaryWriter(gen_log_dir)

mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0,
    debug=False)
mt.load_state_dict(torch.load(join(args.model_dir, args.ckpt)))
mt.test()
mt.cuda()
#%%
## %%time
if condition_file is not None:
    print("use condition file %s, first %d notes" % (condition_file, args.condition_len))
    inputs = np.array([encode_midi(condition_file)[:args.condition_len]])
    print(inputs[0])
else:
    inputs = np.array([[24, 28, 31]])
inputs = torch.from_numpy(inputs).cuda()
with torch.no_grad():
    result = mt(inputs, 2048, gen_summary_writer)
# mid = decode_midi(result.cpu(), file_path=None) #
mid = decode_midi(result, file_path='result/generated_mod%s_cond%s.mid' % (args.ckpt.split('.')[0], condition_fn))
gen_summary_writer.close()
ps_roll = mid.get_piano_roll()
figh = plot_piano_roll(ps_roll)
figh.savefig('result/generated_mod%s_cond%s.jpg' % (args.ckpt.split('.')[0], condition_fn))
#%%
# print(config.condition_file)

# inputs = torch.from_numpy(inputs)
# result = mt(inputs, config.length, gen_summary_writer)
# encode_midi('E:\Datasets\ecomp-midi\Ali03.MID')[:500]

# mid = decode_midi(result, file_path=None)#
# decode_midi(result, file_path=config.save_path)



# figh = plt.figure(figsize=[dur / 10 * xscale,5])
# ax = figh.add_subplot(111)
# plt.hlines(octv_ticks, plt.xlim()[0], plt.xlim()[1], alpha=0.14)
# plt.xlim([mint, maxt])
# plt.yticks(octv_ticks, [M2.pitch.Pitch(p).nameWithOctave for p in octv_ticks])
# mid = encode_midi(cond_file)
# mid = decode_midi(encode_midi(cond_file), file_path=None)#