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
decode_midi(result, file_path='result/generated_mod%s_cond%s.mid' % (args.ckpt.split('.')[0], condition_fn))
gen_summary_writer.close()
#%%
# print(config.condition_file)

# inputs = torch.from_numpy(inputs)
# result = mt(inputs, config.length, gen_summary_writer)
# encode_midi('E:\Datasets\ecomp-midi\Ali03.MID')[:500]

# mid = decode_midi(result, file_path=None)#
# decode_midi(result, file_path=config.save_path)


