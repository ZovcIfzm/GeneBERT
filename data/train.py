


parser = argparse.ArgumentParser(description='DeepDiff')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--model_name', type=str, default='raw_d', help='DeepDiff variation')
parser.add_argument('--clip', type=float, default=1,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=90, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=10, help='')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout) if n_layers LSTM > 1')
parser.add_argument('--save_root', type=str, default='./Results/', help='where to save')
parser.add_argument('--data_root', type=str, default='./data/', help='data location')
parser.add_argument('--gpuid', type=int, default=0, help='CUDA gpu')
parser.add_argument('--gpu', type=int, default=0, help='CUDA gpu')
parser.add_argument('--n_hms', type=int, default=5, help='number of histone modifications')
parser.add_argument('--n_bins', type=int, default=200, help='number of bins')
parser.add_argument('--bin_rnn_size', type=int, default=32, help='bin rnn size')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
parser.add_argument('--unidirectional', action='store_true', help='bidirectional/undirectional LSTM')
parser.add_argument('--save_attention_maps',action='store_true', help='set to save validation beta attention maps')
parser.add_argument('--attentionfilename', type=str, default='beta_attention.txt', help='where to save attnetion maps')
parser.add_argument('--test_on_saved_model',action='store_true', help='only test on saved model')
args = parser.parse_args()

torch.manual_seed(1)



model_name = ''
model_name += (args.cell_1)+('_')+(args.cell_2)+('_')

model_name+=args.model_name

print('the model name: ',model_name)
args.data_root+=''
args.save_root+=''
args.dataset=args.cell_1+('_')+args.cell_2
args.data_root = os.path.join(args.data_root)
print('loading data from:  ',args.data_root)
args.save_root = os.path.join(args.save_root,args.dataset)
print('saving results in  from: ',args.save_root)
model_dir = os.path.join(args.save_root,model_name)
if not os.path.exists(model_dir):
	os.makedirs(model_dir)
attentionmapfile=model_dir+'/'+args.attentionfilename
print('==>processing data')
Train,Valid,Test = data.load_data(args)
