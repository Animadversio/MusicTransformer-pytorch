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
# from tensorboardX import SummaryWriter


# parser = custom.get_argument_parser()
# args = parser.parse_args()
# config.load(args.model_dir, args.configs, initialize=True)
# config.load('generate.yml')
# check cuda
if torch.cuda.is_available():
    config.device = torch.device('cuda')
else:
    config.device = torch.device('cpu')

current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
gen_log_dir = 'logs/mt_decoder/generate_'+current_time+'/generate'
gen_summary_writer = SummaryWriter(gen_log_dir)

# import param
# mt = MusicTransformer(
#     embedding_dim=param.embedding_dim,
#     vocab_size=param.vocab_size,
#     num_layer=param.num_attention_layer,
#     max_seq=param.max_seq,
#     dropout=0,
#     debug=False)

mt = MusicTransformer(
    embedding_dim=config.embedding_dim,
    vocab_size=config.vocab_size,
    num_layer=config.num_layers,
    max_seq=config.max_seq,
    dropout=0,
    debug=False)
# mt.load_state_dict(torch.load(args.model_dir+'/final.pth'))
mt.test()

# def model_size_summary(model):
#     param_num = 0
#     for param in model.parameters():
#         param_num += np.prod(list(param.shape))
#         print(param.shape, "num %d" % np.prod(list(param.shape)))
#     print(param_num, " in total, %.2fmb"%(param_num * 4 / 1024**2))
#
# model_size_summary(mt)
mt.cuda()
#%%
## %%time
inputs = np.array([[24, 28, 31]])
inputs = torch.from_numpy(inputs).cuda()
result = mt(inputs, 1024, gen_summary_writer)
# for i in result:
#     print(i)
mid = decode_midi(result, file_path=None)#
decode_midi(result, file_path='result/generated.mid')
#%%
# print(config.condition_file)
# if config.condition_file is not None:
#     inputs = np.array([encode_midi('dataset/midi/BENABD10.mid')[:500]])
# else:
#     inputs = np.array([[24, 28, 31]])
# inputs = torch.from_numpy(inputs)
# result = mt(inputs, config.length, gen_summary_writer)


mid = decode_midi(result, file_path=None)#
# decode_midi(result, file_path=config.save_path)

gen_summary_writer.close()
