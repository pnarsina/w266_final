import torch

# Setting Vocabulary related parameters
class VocabParameters:
# Original settings
#  embd_dim = 300
    def __init__(self):
        self.data_dir = "./dataset/tacred"
        self.vocab_dir = "./dataset/vocab"
        self.glove_dir = "./dataset/glove"
        self.emb_dim  = 300
        self.vocab_file = "/vocab.pkl"
        self.embed_file = "/embedding.npy"
        self.glove_text_file = "glove.840B.300d.txt"
        self.lower = True 
        self.min_freq =0
        

# Setting different training parameters
class TrainingParameters:
    
# Original settings 
# num_epoch = 30, hidden_dims = 200, attn_dim= 200
    def __init__(self):
        self.ner_dim = 30
        self.pos_dim = 30
        self.hidden_dim =200
        self.num_layers = 2
        self.dropout = 0.5
        self.word_dropout = 0.04
        self.topn = 1e10
        self.lower_dest = 'lower'
        self.lower_action = 'store_true'
        self.no_lower_dest = 'lower'
        self.no_lower_action = 'store_false'
        self.lower = False

        self.attn_dest = 'attn'
        self.attn_action = 'store_true'
        self.no_attn_dest = 'attn'
        self.no_attn_action = 'store_false'
        self.attn = True
        self.attn_dim = 200
        self.pe_dim = 30

#       Optimizer related parameters
        self.lr = 1.0
        self.lr_decay = 0.9
        self.optim = 'sgd'
#       Original model has default of 30, reduced 2 for testing purposes.
        self.num_epoch = 30
        self.batch_size = 50
        self.max_grad_norm = 5
        self.log_step = 20
        self.log = 'logs.txt'
        self.save_epoch = 5
        self.save_dir = './save_models'
        self.id = '00'
        self.info = ''
        
#  CPU / GPU
        self.seed = 1234
        self.cuda = torch.cuda.is_available()
        self.cpu_action = 'store_true'
        self.cpu = True
        

class EvalParameters:
    def __init__(self):
        self.model_dir = "save_models/00"
        self.model = "best.model.pt"
        self.dataset = "test"
        self.out = ""
        self.data_dir = "dataset/tacred"
        self.seed = 1234
        self.cuda = torch.cuda.is_available
        self.cpu_action = "store_true"
        self.cpu = True
                
        