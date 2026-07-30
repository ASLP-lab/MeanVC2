from __future__ import annotations

import os
import gc
from tqdm import tqdm
import wandb

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from src.dataset.npy_dataset import NpyDataset

from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from ema_pytorch import EMA

from src.model import CFM, DiT
from src.model import MeanFlow
from src.model.utils import exists, default, plot_spectrogram, optimized_scale
from src.eval.verification import init_model
from src.eval.cer import process_one, load_zh_model
import zhconv
import torchaudio
import time

# trainer_new: simplified validation — only average SSMI and average CER


class Trainer:
    def __init__(
        self,
        model: DiT,
        args,
        epochs,
        learning_rate,
        # dataloader,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        last_per_steps=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        reset_lr: bool = False,
        grad_ckpt: bool = False,
        # --- new: val set config ---
        val_spks_file: str = None,
        val_bn_file: str = None,
        val_text_path: str = None,
        val_prompt_dir: str = None,
    ):
        self.args = args
        self.expname = wandb_run_name

        # --- val set config (fallback to args or original paths) ---
        self.val_spks_file = val_spks_file or getattr(args, "val_spks_file", None)
        self.val_bn_file = val_bn_file or getattr(args, "val_bn_file", None)
        self.val_text_path = val_text_path or getattr(args, "val_text_path", None)
        self.val_prompt_dir = val_prompt_dir or getattr(args, "val_prompt_dir", None)

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False,)

        logger = "wandb" if wandb.api.api_key else None
        # logger = None
        print(f"Using logger: {logger}")
        self.accelerator = Accelerator(
            log_with=logger,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        if logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "max_samples": self.args.max_len,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                },
            )

        self.precision = self.accelerator.state.mixed_precision
        self.precision = self.precision.replace("no", "fp32")
        print("!!!!!!!!!!!!!!!!!", self.precision)

        self.model = model

        self.args.flow_ratio

        self.meanflow = MeanFlow(
            flow_ratio=self.args.flow_ratio,
            time_dist=['lognorm', -0.4, 1.0],
            cfg_ratio=self.args.cfg_ratio,
            cfg_scale=self.args.cfg_scale,
            cfg_uncond='u',
            p=self.args.p,
        )

        # self.model = torch.compile(model)

        # self.dataloader = dataloader

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)

            self.ema_model.to(self.accelerator.device)
            if self.accelerator.state.distributed_type in ["DEEPSPEED", "FSDP"]:
                self.ema_model.half()

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_e2-tts")

        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.reset_lr = reset_lr

        self.grad_ckpt = grad_ckpt

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        # self.optimizer = FusedAdam(model.parameters(), lr=learning_rate)

        # self.model = torch.compile(self.model)
        if self.accelerator.state.distributed_type == "DEEPSPEED":
            self.accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size

        self.get_dataloader()
        self.get_scheduler()
        self.start_step = self.load_checkpoint()
        # self.get_constant_scheduler()

        self.model, self.optimizer, self.scheduler, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, self.train_dataloader
        )

    def get_scheduler(self):
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        total_steps = len(self.train_dataloader) * self.epochs / self.grad_accumulation_steps
        # total_steps = 5000 * self.accelerator.num_processes
        total_steps = 1000000 * self.accelerator.num_processes
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.05, total_iters=decay_steps)
        # constant_scheduler = ConstantLR(self.optimizer, factor=1, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )

    def get_constant_scheduler(self):
        total_steps = len(self.train_dataloader) * self.epochs / self.grad_accumulation_steps
        self.scheduler = ConstantLR(self.optimizer, factor=1, total_iters=total_steps)

    def get_dataloader(self):
        dataset = NpyDataset(
            filelist_path=self.args.dataset_path,
            feature_list=self.args.feature_list,
            additional_feature_list=self.args.additional_feature_list,
            feature_pad_values=self.args.feature_pad_values,
            max_len=self.args.max_len,
            precision=self.precision,
        )

        self.train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=NpyDataset.custom_collate_fn,
            persistent_workers=self.args.num_workers > 0,
        )

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at step {step}")
            else:
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not os.listdir(self.checkpoint_path)
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            latest_checkpoint = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")],
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )[-1]
        # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location="cpu")

        if self.is_main:
            # del(checkpoint["ema_model_state_dict"]["ema_model.transformer.text_embed.text_embed.weight"])
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False)

        # if "step" in checkpoint:
        if not self.reset_lr:
            # del(checkpoint["model_state_dict"]["transformer.text_embed.text_embed.weight"])
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler and not self.reset_lr:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            step = 0

        del checkpoint
        gc.collect()
        print("step", step)
        return step

    # ------------------------------------------------------------------
    #  Simplified validate:  only average SSMI + average CER
    # ------------------------------------------------------------------
    # ----------------------------------------------------------------
    #  Inference helper: run chunked flow-matching for one (spk, bn)
    # ----------------------------------------------------------------
    def _infer_one(self, model, bn, spk_emb, time_points, steps, chunk_size, block_size):
        bn = bn.unsqueeze(0)
        bn = bn.transpose(1, 2)
        bn_interpolate = torch.nn.functional.interpolate(
            bn, size=int(bn.shape[2] * 4), mode='linear', align_corners=True
        )
        bn = bn_interpolate.transpose(1, 2)

        seq_len = bn.shape[1]
        num_chunks = seq_len // chunk_size
        if seq_len % chunk_size != 0:
            num_chunks += 1

        x_pred_collect = []
        offset = 0
        noise = torch.randn(seq_len, 80, device=self.model.device, dtype=bn.dtype).unsqueeze(0)

        for chunk_id in range(num_chunks):
            start = chunk_id * chunk_size
            mid = min(start + chunk_size, seq_len)
            end = min(mid + block_size, seq_len)
            curr_len = mid - start

            bn_hist = bn[:, :start]
            bn_curr = bn[:, start:mid]
            bn_fut = bn[:, mid:end]
            cond_in = (
                torch.cat([bn_hist, bn_curr, bn_fut], dim=1)
                if start > 0
                else torch.cat([bn_curr, bn_fut], dim=1)
            )

            x = noise[:, :cond_in.shape[1]]

            for i_step in range(steps):
                t = time_points[i_step]
                r = time_points[i_step + 1]
                t_tensor = torch.full((x.size(0),), t, device=x.device)
                r_tensor = torch.full((x.size(0),), r, device=x.device)
                with torch.inference_mode():
                    u = model(
                        x, t_tensor, r_tensor,
                        cache=None,
                        cond=cond_in,
                        spks=spk_emb,
                        offset=offset,
                        is_inference=True,
                    )
                    x = x - (t - r) * u

            curr_start = bn_hist.shape[1] if start > 0 else 0
            x_emit = x[:, curr_start : curr_start + curr_len]
            offset += curr_len
            x_pred_collect.append(x_emit)

        x_pred = torch.cat(x_pred_collect, dim=1)
        return x_pred

    def _validate_one_lang(self, spks_file, bn_file, text_path, spk_prefix, use_dir_join, step, model, vocos, sv_model, asr_model, time_points, steps, chunk_size, block_size):
        """Validate one language test set.  Returns {spk: ssmi_mean}, {spk: wer}."""
        utt2txt = {}
        text_all = ''
        for line in open(text_path, 'r').readlines():
            utt, txt = line.strip().split('|')
            utt2txt[utt] = txt
            text_all += utt2txt[utt]

        if use_dir_join:
            spks_dir = os.path.dirname(spks_file)
            bns_dir = os.path.dirname(bn_file)
            spks = [os.path.join(spks_dir, line.strip()) for line in open(spks_file, "r")]
            bns = [os.path.join(bns_dir, line.strip()) for line in open(bn_file, "r")]
        else:
            spks = [line.strip() for line in open(spks_file, "r")]
            bns = [line.strip() for line in open(bn_file, "r")]

        ssmi_dict = {}
        wer_dict = {}

        with torch.no_grad():
            for spk in spks:
                ssmi_list = []
                transcription_all = ""
                spk_filename = os.path.basename(spk).split(".")[0]

                spk_result_dir = os.path.join(self.args.result_dir, self.expname, str(step), spk_prefix + spk_filename)
                os.makedirs(spk_result_dir, exist_ok=True)

                spk_emb = np.load(spk)
                spk_emb = torch.from_numpy(spk_emb).to(self.model.device)
                if len(spk_emb.shape) == 1:
                    spk_emb = spk_emb.unsqueeze(0)

                for bn_path in bns:
                    bn = torch.from_numpy(np.load(bn_path)).to(self.model.device)
                    x_pred = self._infer_one(model, bn, spk_emb, time_points, steps, chunk_size, block_size)

                    base_filename = os.path.basename(bn_path).split(".")[0]
                    mel_output_path = os.path.join(spk_result_dir, base_filename + ".npy")
                    np.save(mel_output_path, x_pred.cpu().numpy())

                    mel = x_pred.transpose(1, 2)
                    mel = (mel + 1) / 2
                    y_g_hat = vocos.decode(mel)
                    wav_output_path = os.path.join(spk_result_dir, base_filename + ".wav")
                    torchaudio.save(wav_output_path, y_g_hat.cpu(), 16000)

                    time.sleep(1)

                    # SSMI
                    with torch.no_grad():
                        emb = sv_model(y_g_hat)
                    ssmi = F.cosine_similarity(spk_emb, emb)
                    ssmi_list.append(ssmi.item())

                    # CER
                    try:
                        res = asr_model.generate(input=wav_output_path, batch_size_s=300)
                        transcription = res[0]["text"]
                        transcription = zhconv.convert(transcription, 'zh-cn')
                    except Exception:
                        import traceback
                        print(f"Error processing {wav_output_path}:")
                        traceback.print_exc()
                        transcription = ""

                    transcription_all += transcription

                wer = process_one(transcription_all, text_all)
                wer_dict[spk_filename] = wer
                ssmi_dict[spk_filename] = np.mean(ssmi_list)

        return ssmi_dict, wer_dict

    def validate(self, step):
        self.model.eval()
        model = self.accelerator.unwrap_model(self.model)

        vocos = torch.jit.load("ckpts/vocos/vocos.pt").to(self.model.device)

        sv_model = init_model(
            'wavlm_large',
            'preprocess/ckpts/wavlm_large_finetune.pth'
        )
        sv_model.eval()
        sv_model.to(self.model.device)

        asr_model = load_zh_model(str(self.model.device))

        steps = self.args.steps
        chunk_size = self.args.chunk_size
        block_size = self.args.block_size
        time_points = torch.linspace(1.0, 0.0, steps + 1, device=self.model.device)

        # # --- Chinese validation ---
        zh_spks_file = "val/zh/spks.lst"
        zh_bn_file = "val/zh/bn.scp"
        zh_text_path = "val/zh/text.txt"
        print(f"[validate ZH] spks={zh_spks_file}, bn={zh_bn_file}, text={zh_text_path}")
        zh_ssmi_dict, zh_wer_dict = self._validate_one_lang(
            zh_spks_file, zh_bn_file, zh_text_path, spk_prefix="zh_", use_dir_join=False,
            step=step, model=model, vocos=vocos, sv_model=sv_model, asr_model=asr_model,
            time_points=time_points, steps=steps, chunk_size=chunk_size, block_size=block_size,
        )

        # --- English validation ---
        en_spks_file = "val/en/spks.lst"
        en_bn_file = "val/en/bn.scp"
        en_text_path = "val/en/text.txt"
        print(f"[validate EN] spks={en_spks_file}, bn={en_bn_file}, text={en_text_path}")
        en_ssmi_dict, en_wer_dict = self._validate_one_lang(
            en_spks_file, en_bn_file, en_text_path, spk_prefix="en_", use_dir_join=True,
            step=step, model=model, vocos=vocos, sv_model=sv_model, asr_model=asr_model,
            time_points=time_points, steps=steps, chunk_size=chunk_size, block_size=block_size,
        )

        # --- log per-speaker ---
        for spk in zh_ssmi_dict:
            self.accelerator.log({f"zh_ssmi_{spk}": zh_ssmi_dict[spk], f"zh_wer_{spk}": zh_wer_dict[spk]}, step=step)
        for spk in en_ssmi_dict:
            self.accelerator.log({f"en_ssmi_{spk}": en_ssmi_dict[spk], f"en_wer_{spk}": en_wer_dict[spk]}, step=step)

        # --- log averages ---
        zh_avg_ssmi = np.mean(list(zh_ssmi_dict.values())) if zh_ssmi_dict else 0.0
        zh_avg_cer = np.mean(list(zh_wer_dict.values())) if zh_wer_dict else 0.0
        en_avg_ssmi = np.mean(list(en_ssmi_dict.values())) if en_ssmi_dict else 0.0
        en_avg_cer = np.mean(list(en_wer_dict.values())) if en_wer_dict else 0.0

        print(f"[validate] step={step}  zh_avg_ssmi={zh_avg_ssmi:.4f}  zh_avg_cer={zh_avg_cer:.4f}  en_avg_ssmi={en_avg_ssmi:.4f}  en_avg_cer={en_avg_cer:.4f}")
        # print(f"[validate] step={step}  en_avg_ssmi={en_avg_ssmi:.4f}  en_avg_cer={en_avg_cer:.4f}")
        self.accelerator.log({
            "val_zh_avg_ssmi": zh_avg_ssmi, "val_zh_avg_cer": zh_avg_cer,
            "val_en_avg_ssmi": en_avg_ssmi, "val_en_avg_cer": en_avg_cer,
        }, step=step)

        self.model.train()
        del sv_model
        del asr_model
        torch.cuda.empty_cache()

    def train(self, resumable_with_seed: int = None):
        train_dataloader = self.train_dataloader

        start_step = self.start_step
        global_step = start_step

        self.model.train()

        if resumable_with_seed > 0:
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if resumable_with_seed > 0 and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                    smoothing=0.15,
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    smoothing=0.15,
                )
            for batch in progress_bar:
                with self.accelerator.accumulate(self.model):
                    features = {}
                    for feature_name in self.args.feature_list + self.args.additional_feature_list:
                        features[feature_name] = batch[feature_name]

                    diff_loss, mse_val = self.meanflow.loss(
                        self.model,
                        x=features["mel"],
                        bn=features["bn"],
                        spks=features["xvector"],
                        # prompts=features["prompt"],
                        inputs_length=features["inputs_length"],
                    )

                    self.accelerator.backward(diff_loss)

                    grad_norm = None
                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.is_main:
                    self.ema_model.update()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    log_data = {
                        "loss": diff_loss.item(),
                        "mse_val": mse_val.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    if grad_norm is not None:
                        log_data["grad_norm"] = round(grad_norm.item(), 2)

                    self.accelerator.log(log_data, step=global_step)

                progress_bar.set_postfix(step=str(global_step), loss=diff_loss.item(), mse_val=mse_val.item())

                if global_step % 10000 == 0:
                    if self.accelerator.is_main_process:
                        try:
                            self.validate(global_step)
                        except Exception as e:
                            print(f"ERROR: {e}")
                    self.accelerator.wait_for_everyone()

                if global_step % (self.save_per_updates * self.grad_accumulation_steps) == 0:
                    self.save_checkpoint(global_step)

                if global_step % self.last_per_steps == 0:
                    self.save_checkpoint(global_step, last=True)

        self.save_checkpoint(global_step, last=True)

        self.accelerator.end_training()
