import sys, os
from tqdm import tqdm
from jiwer import compute_measures
from zhon.hanzi import punctuation
import string
import numpy as np
import soundfile as sf
import scipy
import zhconv
from funasr import AutoModel
import glob

punctuation_all = punctuation + string.punctuation

def load_zh_model(device):

    model = AutoModel(
        model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        model_hub="ms",
        disable_pbar=True,
        disable_update=True,
        device=device
    )

    return model

def process_one(hypo, truth):
    for x in punctuation_all:
        if x == '\'':
            continue
        truth = truth.replace(x, '')
        hypo = hypo.replace(x, '')

    truth = truth.replace('  ', ' ')
    hypo = hypo.replace('  ', ' ')


    truth = " ".join([x for x in truth])
    hypo = " ".join([x for x in hypo])

    measures = compute_measures(truth, hypo)
    wer = measures["wer"]
    return wer

