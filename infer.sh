#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3

wavpath_file="/data/lichunyou/aishell_data/kaldi_ark_format/test.txt"
model_file="model_out/epoch65_train:0.5679027072161724_val:8.722621902478023.pth.tar"
output_file="test_dataset/aishell_text_infer"
conf="conf/config.conf"
beam_size=5
is_cuda=True

python infer.py                      \
    --recog-conf ${conf}             \
    --model-file ${model_file}       \
    --wav-path-file ${wavpath_file}  \
    --output-txt ${output_file}       \
    --beam-size ${beam_size}         \
    --nbest ${beam_size}             
