import os
import json
import argparse
import glob

from dit_kvcache import DiT

from src.model.utils import load_checkpoint
import numpy as np
import torch
import time
from tqdm import tqdm
import torchaudio
from functools import partial
from multiprocessing import Pool
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.inference_mode()
def inference(model, vocos, bn_path, spk_emb, prompt_mel, chunk_size, steps, device,):
    # block_size = chunk_size
    block_size = 4  # [FIXED] 40ms, hardcoded to support 120ms+40ms asymmetric chunks
    timesteps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    bn = torch.from_numpy(np.load(bn_path)).to(device)
    bn = bn.unsqueeze(0)
    bn = bn.transpose(1, 2)
    bn_interpolate = torch.nn.functional.interpolate(bn, size=int(bn.shape[2] * 4), mode='linear', align_corners=True)
    bn = bn_interpolate.transpose(1, 2)

    seq_len = bn.shape[1]
    num_chunks = seq_len // chunk_size
    if seq_len % chunk_size != 0:
        num_chunks += 1
        
    x_pred_collect = []

    offset = 0
    noise = torch.randn(seq_len, 80, device=device, dtype=bn.dtype).unsqueeze(0) 
    kv_cache = None

    s_t_item = time.time()
    for chunk_id in range(num_chunks):
        start = chunk_id * chunk_size
        mid = min(start + chunk_size, seq_len)
        end = min(mid + block_size, seq_len)
        curr_len = mid - start
        
        bn_curr = bn[:, start:mid]
        bn_fut = bn[:, mid:end]
        # cond_in = torch.cat([bn_hist, bn_curr, bn_fut], dim=1) if start > 0 else torch.cat([bn_curr, bn_fut], dim=1)
        cond_in = torch.cat([bn_curr, bn_fut], dim=1)       # only use current and future frames as condition; history uses kvcache
            
        # x = noise[:, :cond_in.shape[1]]
        x = noise[:, start:end]  
        cfg_mask = torch.ones([x.shape[0]], dtype=torch.bool, device=device)
        for i in range(steps):
            t = timesteps[i]
            r = timesteps[i+1]
            t_tensor = torch.full((x.size(0),), t, device=x.device)
            r_tensor = torch.full((x.size(0),), r, device=x.device)
            with torch.inference_mode():

                u, tmp_kv_cache = model(
                    x,
                    t_tensor,
                    r_tensor,
                    cache=None,
                    cond=cond_in,
                    spks=spk_emb,
                    # prompts=prompt_mel,
                    offset=offset,
                    is_inference=True,
                    kv_cache=kv_cache,
                )
                
                x = x - (t - r) * u
        
        kv_cache = tmp_kv_cache
        x_emit = x[:, :curr_len]  # Directly output the current block's generated results; kvcache handles history
        offset += curr_len
        x_pred_collect.append(x_emit) 

    
    x_pred = torch.cat(x_pred_collect, dim=1)
    mel = x_pred.transpose(1,2)
    mel = (mel + 1) / 2
    # import pdb;pdb.set_trace()
    y_g_hat = vocos.decode(mel)
    # gain = -3.0
    # y_g_hat, _ = torchaudio.sox_effects.apply_effects_tensor(y_g_hat, 16000, [["norm", f"{gain:.2f}"]])
    time_item = time.time() - s_t_item

    return mel, y_g_hat, time_item



def load_model(model_config, ckpt_path, device, use_ema):
    model_cls = DiT
    dit_model = model_cls(**model_config["model"])
    dit_model = dit_model.to(device)
    dit_model = load_checkpoint(dit_model, ckpt_path, device=device, use_ema=use_ema)
    dit_model = dit_model.float()
    dit_model.eval()
    return dit_model

# Global function to avoid pickle issues
def process_utterance_worker(args_tuple):
    """
    Global worker function for multiprocessing.
    args_tuple contains all required parameters.
    """
    (utt, model_config_path, ckpt_path, vocoder_ckpt_path, bn_path, spk_emb_path, 
     prompt_path, chunk_size, steps, output_dir_mel, output_dir_wav, device, use_ema) = args_tuple
    
    try:
        # Load model in each process (loaded only once)
        with open(model_config_path) as f:
            model_config = json.load(f)
        
        dit_model = load_model(model_config, ckpt_path, device, use_ema)
        vocos = torch.jit.load(vocoder_ckpt_path).to(device)
        
        # Process a single utterance
        if "|" in utt:
            tar_utt, source_utt = utt.split("|")
        else:
            tar_utt, source_utt = utt.split("_")
        bn_file_path = os.path.join(bn_path, source_utt + ".npy")
        
        # Load speaker embedding
        spk_emb_file_path = os.path.join(spk_emb_path, tar_utt + ".npy")
        spk_emb = np.load(spk_emb_file_path)
        spk_emb = torch.from_numpy(spk_emb).to(device)
        if len(spk_emb.shape) == 1:
            spk_emb = spk_emb.unsqueeze(0)
            
        # Load prompt
        # prompt_file_path = os.path.join(prompt_path, tar_utt + ".npy")
        # prompt_mel = np.load(prompt_file_path)
        # prompt_mel = torch.from_numpy(prompt_mel).to(device)
        # if len(prompt_mel.shape) < 3:
        #     prompt_mel = prompt_mel.unsqueeze(0)
        # if prompt_mel.shape[1] == 80:
        #     prompt_mel = prompt_mel.transpose(1, 2)
        
        # Run inference
        mel, wav, time_item = inference(dit_model, vocos, bn_file_path, spk_emb, None, chunk_size, steps, device)
        
        # Save results
        mel_output_path = os.path.join(output_dir_mel, utt + ".npy")
        np.save(mel_output_path, mel.cpu().numpy())
        
        wav_output_path = os.path.join(output_dir_wav, utt + ".wav")
        torchaudio.save(wav_output_path, wav.cpu(), 16000)
        
        return utt, time_item, None
        
    except Exception as e:
        return utt, None, str(e)

# Batch processing worker function
def process_batch_worker(args_tuple):
    """
    Batch processing worker function.
    """
    (batch_utts, model_config_path, ckpt_path, vocoder_ckpt_path, bn_path, spk_emb_path, 
     prompt_path, chunk_size, steps, output_dir_mel, output_dir_wav, device, use_ema) = args_tuple
    
    results = []
    dit_model = None
    vocos = None
    
    try:
        # Load models at the start of batch processing (loaded only once per process)
        with open(model_config_path) as f:
            model_config = json.load(f)
        
        dit_model = load_model(model_config, ckpt_path, device, use_ema)
        vocos = torch.jit.load(vocoder_ckpt_path).to(device)
        
        # Process each utterance in the batch
        for utt in batch_utts:
            try:
                if "|" in utt:
                    tar_utt, source_utt = utt.split("|")
                else:
                    tar_utt, source_utt = utt.split("_")
                bn_file_path = os.path.join(bn_path, source_utt + ".npy")
                
                # Load speaker embedding
                spk_emb_file_path = os.path.join(spk_emb_path, tar_utt + ".npy")
                spk_emb = np.load(spk_emb_file_path)
                spk_emb = torch.from_numpy(spk_emb).to(device)
                if len(spk_emb.shape) == 1:
                    spk_emb = spk_emb.unsqueeze(0)
                    
                # Load prompt
                # prompt_file_path = os.path.join(prompt_path, tar_utt + ".npy")
                # prompt_mel = np.load(prompt_file_path)
                # prompt_mel = torch.from_numpy(prompt_mel).to(device)
                # if len(prompt_mel.shape) < 3:
                #     prompt_mel = prompt_mel.unsqueeze(0)
                # if prompt_mel.shape[1] == 80:
                #     prompt_mel = prompt_mel.transpose(1, 2)
                
                # Run inference
                mel, wav, time_item = inference(dit_model, vocos, bn_file_path, spk_emb, None, chunk_size, steps, device)
                
                # Save results
                mel_output_path = os.path.join(output_dir_mel, utt + ".npy")
                np.save(mel_output_path, mel.cpu().numpy())
                
                wav_output_path = os.path.join(output_dir_wav, utt + ".wav")
                torchaudio.save(wav_output_path, wav.cpu(), 16000)
                
                results.append((utt, time_item, None))
                
            except Exception as e:
                results.append((utt, None, str(e)))
                print(f"Error processing {utt}: {e}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        for utt in batch_utts:
            results.append((utt, None, str(e)))
    
    return results

def simple_multiprocessing_inference(args, utts, num_processes=4):
    """Simple multiprocess inference; each process handles one utterance."""
    
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_ema = args.use_ema.lower() == 'true'
    
    output_dir_mel = os.path.join(args.output_dir, "mel")
    os.makedirs(output_dir_mel, exist_ok=True)
    output_dir_wav = os.path.join(args.output_dir, "wav")
    os.makedirs(output_dir_wav, exist_ok=True)
    
    # Prepare all parameters
    worker_args = [
        (utt, args.model_config, args.ckpt_path, args.vocoder_ckpt_path, 
         args.bn_path, args.spk_emb_path, args.prompt_path, 
         args.chunk_size, args.steps, output_dir_mel, output_dir_wav, device, use_ema)
        for utt in utts
    ]
    
    results = []
    with Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap(process_utterance_worker, worker_args), 
                          total=len(utts), desc="Processing utterances"):
            results.append(result)
            utt, time_item, error = result
            if error:
                print(f"Error processing {utt}: {error}")
            elif time_item:
                print(f"Processed {utt} in {time_item:.2f}s")
    
    return results

def batch_multiprocessing_inference(args, utts, num_processes=4, batch_size=10):
    """Batch multiprocess inference."""
    
    # device = 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_ema = args.use_ema.lower() == 'true'
    
    output_dir_mel = os.path.join(args.output_dir, "mel")
    os.makedirs(output_dir_mel, exist_ok=True)
    output_dir_wav = os.path.join(args.output_dir, "wav")
    os.makedirs(output_dir_wav, exist_ok=True)
    
    # Split utterances into batches
    batches = [utts[i:i + batch_size] for i in range(0, len(utts), batch_size)]
    
    # Prepare batch processing parameters
    batch_args = [
        (batch, args.model_config, args.ckpt_path, args.vocoder_ckpt_path, 
         args.bn_path, args.spk_emb_path, args.prompt_path, 
         args.chunk_size, args.steps, output_dir_mel, output_dir_wav, device, use_ema)
        for batch in batches
    ]
    
    all_results = []
    with Pool(processes=num_processes) as pool:
        batch_results = list(tqdm(pool.imap(process_batch_worker, batch_args), 
                                 total=len(batches), desc="Processing batches"))
        
        for batch_result in batch_results:
            all_results.extend(batch_result)
            
        # Print result statistics
        successful = sum(1 for _, time_item, error in all_results if error is None)
        failed = len(all_results) - successful
        print(f"Successfully processed: {successful}, Failed: {failed}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--vocoder-ckpt-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--file-scp', type=str, default=None)
    parser.add_argument('--bn-path', type=str, default=None)
    parser.add_argument('--spk-emb-path', type=str, default=None)
    parser.add_argument('--prompt-path', type=str, default=None)
    parser.add_argument('--chunk-size', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_ema', type=str, default='True')
    parser.add_argument('--method', type=str, default='batch', 
                       choices=['simple', 'batch'],
                       help='Multiprocessing method: simple (one utterance per process), batch')
    parser.add_argument('--num-processes', type=int, default=20, help='Number of processes')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size')

    args = parser.parse_args()
    setup_seed(args.seed)

    with open(args.file_scp) as f:
        utts = [line.strip() for line in f.readlines()]

    print(f"Processing {len(utts)} utterances using {args.method} method...")
    print(f"Number of processes: {args.num_processes}")
    if args.method == 'batch':
        print(f"Batch size: {args.batch_size}")
    
    start_time = time.time()
    
    if args.method == 'simple':
        results = simple_multiprocessing_inference(args, utts, args.num_processes)
    elif args.method == 'batch':
        results = batch_multiprocessing_inference(args, utts, args.num_processes, args.batch_size)
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per utterance: {total_time/len(utts):.2f}s")