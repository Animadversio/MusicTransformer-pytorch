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
from os.path import join

parser = argparse.ArgumentParser(None)
parser.add_argument("-m", "--model_dir", type=str, required=True,
        help="The directory for a trained model is saved.")
parser.add_argument("-k", "--ckpt", type=str, dest="ckpt", default="final.pth",
        help="name of checkpoint file")
parser.add_argument("-c", "--conf", dest="configs", default=["generate.yml"], nargs="*",
        help="A list of configuration items. "
             "An item is a file path or a 'key=value' formatted string. "
             "The type of a value is determined by applying int(), float(), and str() "
             "to it sequencially.")
args = parser.parse_args()
config.load(args.model_dir, args.configs, initialize=True)

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

inputs = np.array([[60, 64, 67]])
inputs = torch.from_numpy(inputs).cuda()
with torch.no_grad():
    result = mt(inputs, 1024, gen_summary_writer)
# mid = decode_midi(result.cpu(), file_path=None) #
decode_midi(result, file_path='result/generated%s.mid'%args.ckpt)
gen_summary_writer.close()
#%%
# print(config.condition_file)
# if config.condition_file is not None:
#     inputs = np.array([encode_midi('dataset/midi/BENABD10.mid')[:500]])
# else:
#     inputs = np.array([[24, 28, 31]])
# inputs = torch.from_numpy(inputs)
# result = mt(inputs, config.length, gen_summary_writer)


# mid = decode_midi(result, file_path=None)#
# decode_midi(result, file_path=config.save_path)


