from configparser import ConfigParser

class TrainerConf:
    def __init__(self):
        self.datasets = DatasetsConf()
        self.audio_feature = Audio_featureConf()
        self.text_feature = Text_featureConf()
        self.training = TrainingConf()
        self.transformer = TransformerConf()
    
    def load(self,config_name):
        conf_parser = ConfigParser()
        ret = conf_parser.read(config_name, encoding='utf-8')

        self.datasets.load(conf_parser)
        self.audio_feature.load(conf_parser)
        self.text_feature.load(conf_parser)
        self.training.load(conf_parser)
        self.transformer.load(conf_parser)


class DatasetsConf:
    def __init__(self):
        self.train_wav2text_file = ""  
        self.valid_wav2text_file = ""
        self.batch_seconds       = 300
    def load(self,conf_parser):
        self.train_wav2text_file = conf_parser.get("datasets","train_wav2text_file")
        self.valid_wav2text_file = conf_parser.get("datasets","valid_wav2text_file")
        self.batch_seconds       = conf_parser.getint("datasets","batch_seconds")

class Audio_featureConf:
    def __init__(self):
        self.cmvn_npy_file          = "conf/cmvn.txt"
        self.fbank_dim              = 40
        self.sampling_rate          = 16000
        self.low_frame_rate_stack   = 25
        self.low_frame_rate_stride  = 10
    def load(self,conf_parser):
        self.cmvn_npy_file          = conf_parser.get("audio_feature","cmvn_npy_file")
        self.fbank_dim              = conf_parser.getint("audio_feature","fbank_dim")
        self.sampling_rate          = conf_parser.getint("audio_feature","sampling_rate")
        self.low_frame_rate_stack   = conf_parser.getint("audio_feature","low_frame_rate_stack")
        self.low_frame_rate_stride  = conf_parser.getint("audio_feature","low_frame_rate_stride")

class Text_featureConf:
    def __init__(self):
        self.char2token_file = "conf/labels.json"
    def load(self,conf_parser):
        self.char2token_file = conf_parser.get("text_feature","char2token_file")

class TrainingConf:
    def __init__(self):
        self.continue_from   = ""  
        self.model_output    = "model_out"
        self.max_epoches     =  150
        self.lr_k            =  0.2
        self.warmup_steps    =  4000
        self.label_smoothing = 0.1
    def load(self,conf_parser):
        self.continue_from   = conf_parser.get("training","continue_from")
        self.model_output    = conf_parser.get("training","model_output")
        self.max_epoches     = conf_parser.getint("training","max_epoches")
        self.lr_k            = conf_parser.getfloat("training","lr_k")
        self.warmup_steps    = conf_parser.getfloat("training","warmup_steps")
        self.label_smoothing = conf_parser.getfloat("training","label_smoothing")

class TransformerConf:
    def __init__(self):
        self.n_layers_enc                 =    4
        self.n_head                       =    8
        self.d_k                          =    64
        self.d_v                          =    64
        self.d_model                      =    512
        self.d_inner                      =    2048
        self.dropout                      =    0.1
        self.pe_maxlen                    =    5000
        self.n_layers_dec                 =    4
        self.tgt_emb_prj_weight_sharing   =    1
    def load(self,conf_parser):
        self.n_layers_enc                 =    conf_parser.getint("transformer","n_layers_enc")
        self.n_head                       =    conf_parser.getint("transformer","n_head")
        self.d_k                          =    conf_parser.getint("transformer","d_k")
        self.d_v                          =    conf_parser.getint("transformer","d_v")
        self.d_model                      =    conf_parser.getint("transformer","d_model")
        self.d_inner                      =    conf_parser.getint("transformer","d_inner")
        self.dropout                      =    conf_parser.getfloat("transformer","dropout")
        self.pe_maxlen                    =    conf_parser.getint("transformer","pe_maxlen")
        self.n_layers_dec                 =    conf_parser.getint("transformer","n_layers_dec")
        self.tgt_emb_prj_weight_sharing   =    conf_parser.getint("transformer","tgt_emb_prj_weight_sharing")
