import sys
import os
from torch._C import import_ir_module
import torch
import torch.nn as nn

from tools.params_parse import TrainerConf
from tools.char_token import CharTokenizer
from tools.feature_computer import FeatureComputer
from loader.data_loader import AudioTextDataset,AudioTextDataLoader
from model.encoder import Encoder
from model.decoder import Decoder
from model.transformer import Transformer
from model.optimizer import TransformerOptimizer
from model.loss import cal_performance
from model.pad_mask_utils import IGNORE_ID


def train(cfg):
    
    #加载字典
    labels = CharTokenizer(cfg.text_feature.char2token_file)
    
    #特征计算
    feature = FeatureComputer(cfg)

    #dataset,dataloader
    train_dataset = AudioTextDataset(manifest_filepath = cfg.datasets.train_wav2text_file,  
                                     batch_seconds     = cfg.datasets.batch_seconds          
                                     )  
    test_dataset = AudioTextDataset(manifest_filepath = cfg.datasets.valid_wav2text_file,  
                                     batch_seconds    = cfg.datasets.batch_seconds          
                                    )  
    
    train_loader = AudioTextDataLoader(dataset          = train_dataset,
                                       text_tokenizer   = labels,
                                       feature_computer = feature
                                       )   
    test_loader = AudioTextDataLoader(dataset          = test_dataset,
                                       text_tokenizer   = labels,
                                       feature_computer = feature
                                       )
    
    sos_id = labels.get_sos_token()               #"<sos>"的id
    eos_id = labels.get_eos_token()               #"<eos>"的id
    vocab_size = labels.get_vocab_size()          #字典大小

    
    encoder = Encoder(feature.get_feature_dim(),      
                      cfg.transformer.n_layers_enc,   
                      cfg.transformer.n_head,         
                      cfg.transformer.d_k,            
                      cfg.transformer.d_v,            
                      cfg.transformer.d_model,        
                      cfg.transformer.d_inner,        
                      dropout=cfg.transformer.dropout,
                      pe_maxlen=cfg.transformer.pe_maxlen
                      ) 
    decoder = Decoder(sos_id,
                      eos_id,
                      vocab_size,
                      cfg.transformer.d_model,       
                      cfg.transformer.n_layers_dec, 
                      cfg.transformer.n_head,       
                      cfg.transformer.d_k,           
                      cfg.transformer.d_v,           
                      cfg.transformer.d_model,       
                      cfg.transformer.d_inner,       
                      dropout=cfg.transformer.dropout,
                      tgt_emb_prj_weight_sharing=cfg.transformer.tgt_emb_prj_weight_sharing, #1
                      pe_maxlen=cfg.transformer.pe_maxlen
                     ) 

    model = Transformer(encoder, decoder)
    
    #continue训练:
    if cfg.training.continue_from != "":
        print("continue train from: {}".format(cfg.training.continue_from))
        model = model.load_model(cfg.training.continue_from)
    
    gpu_num = torch.cuda.device_count()
    device = device = torch.device("cuda" if gpu_num>0 else "cpu")
    model = model.to(device)

    #分布式
    if gpu_num > 1:
        model = torch.nn.DataParallel(model, list(range(gpu_num)))
    
    optimizer = TransformerOptimizer(
                    torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
                    cfg.training.lr_k,                    
                    cfg.transformer.d_model,                
                    cfg.training.warmup_steps
                    )              
    
     #创建模型保存目录
    if not os.path.exists(cfg.training.model_output):
        os.makedirs(cfg.training.model_output)


    for epoch in range(cfg.training.max_epoches):

        #训练
        model.train()   
        tr_avg_loss,train_cer = run_one_epoch(cfg, epoch,model,optimizer,train_loader,device,False)
        print('-' * 85)
        print('Train Summary | End of Epoch {0} | Train Loss {1:.3f}'.format(epoch + 1, tr_avg_loss))
        print('-' * 85)
        
        #验证
        print('Cross validation...')
        model.eval()
        with torch.no_grad():
            val_loss,val_cer = run_one_epoch(cfg, epoch,model,optimizer,test_loader,device,True)
        print('-' * 85)
        print('Valid Summary | End of Epoch {0} | Valid Loss {1:.3f}'.format(epoch + 1, val_loss))
        print('-' * 85)

        #保存模型
        file_path = os.path.join(
                    cfg.training.model_output, 'epoch{}_train:{}_val:{}.pth.tar'.format(epoch + 1,train_cer,val_cer))

        if isinstance(model,nn.DataParallel):
            torch.save(model.module.serialize(model.module, optimizer, epoch + 1,tr_loss=tr_avg_loss, cv_loss=val_loss),
                       file_path)
        else:
            torch.save(model.serialize(model, optimizer, epoch + 1,tr_loss=tr_avg_loss, cv_loss=val_loss),
                       file_path)
        

def run_one_epoch(cfg, epoch, model, optimizer ,data_loader,device,cross_valid=False):

    total_loss = 0
    total_iters = len(data_loader)
    total_word = 0
    total_correct = 0

    for i, (data) in enumerate(data_loader):
        padded_input, input_lengths, padded_targets, wav_paths = data

        padded_input = padded_input.to(device)
        input_lengths = input_lengths.to(device)
        padded_targets = padded_targets.to(device)

        pred, gold = model(padded_input, input_lengths, padded_targets)         
        loss, n_correct = cal_performance(pred, gold,
                                            smoothing=cfg.training.label_smoothing)
        if not cross_valid:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        n_correct-=1
        n_word-=1
        

        if i % 100 == 0:
            print('Epoch {} | Iter {}/{} | EpAvg Loss {:.3f} | '
                                'Current Loss {:.4f} | Train WER {:.1f}% '.format(
                                epoch + 1, i + 1, total_iters, total_loss / (i + 1),
                                loss.item(), 100 * (1 - n_correct / n_word)))
        
        total_word += n_word
        total_correct += n_correct

    total_cer = 100 * (1 - total_correct / total_word)
    print('train_cer {} '.format(total_cer))

    return total_loss / (i + 1),total_cer




if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("param error!")
        sys.exit(1)

    #配置文件，加载配置文件
    config_filename = sys.argv[1]

    cfg = TrainerConf()             #
    cfg.load(config_filename)

    train(cfg)



