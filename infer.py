from shutil import ignore_patterns
import sys
import os
import numpy as np
import argparse
from tqdm import tqdm
import torch

from tools.char_token import CharTokenizer
from tools.feature_computer import FeatureComputer
from tools.params_parse import TrainerConf
from loader.data_loader import AudioTextDataset,AudioTextDataLoader
from model.transformer import Transformer
from model.beam_decode import Recognizer


def infer(cfg, model_file, input_file, output_file, beam_size, nbest):
    #加载字典
    labels = CharTokenizer(cfg.text_feature.char2token_file)
    #特征计算
    feature = FeatureComputer(cfg)

    dataset = AudioTextDataset(manifest_filepath = input_file,  
                               batch_seconds     = cfg.datasets.batch_seconds,
                               for_training      = False        
                               )  
    dataloader = AudioTextDataLoader(dataset          = dataset,
                                     text_tokenizer   = labels,
                                     feature_computer = feature
                                     )   
    
    
    gpu_num = torch.cuda.device_count()
    device = torch.device("cuda" if gpu_num>0 else "cpu")

    model = Transformer.load_model(model_file)
    model = model.to(device)
    model.eval()

    
    recognizer = Recognizer(labels,feature,model)

    

    all_top1_results = {}
    for data in tqdm(dataloader):
        padded_input, input_lengths, padded_targets, wav_paths = data

        padded_input = padded_input.to(device)
        input_lengths = input_lengths.to(device)

        for i in range(len(padded_input)):
            this_input = padded_input[i]
            this_length = input_lengths[i: i + 1]
            this_path = wav_paths[i]

            result = recognizer.predict_given_feature(this_input, this_length, beam_size, nbest)

            if result.ret_flag == 0 and len(result.result_item_list):
                result_item = result.result_item_list[0]
                print('{}\t{:.3f}\t{}'.format(this_path, result_item.score, result_item.text))
                all_top1_results[this_path] = (result_item.score, result_item.text)
    

    with open(output_file, 'w') as f:
        for wav_path, _, _, in dataset.wav_text_duration_list:
            if wav_path not in all_top1_results:
                continue
            top1_conf, top1_text = all_top1_results[wav_path]
            line = '{}\t{}\t{:.3f}\n'.format(wav_path, top1_text, top1_conf)
            f.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ASR  Inference')

    parser.add_argument('--recog-conf', type=str,default="")
    parser.add_argument("--model-file",type= str)
    parser.add_argument('--wav-path-file', type=str)
    parser.add_argument('--output-txt', type=str)
    parser.add_argument('--beam-size', type=int, default=2)
    parser.add_argument('--nbest', type=int, default=2)

    args = parser.parse_args()

    cfg = TrainerConf()
    cfg.load(args.recog_conf)

    infer(cfg, args.model_file, args.wav_path_file , args.output_txt, args.beam_size, args.nbest)
    
