import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from .save_cfg import SaveCfg
from .train_cfg import TrainerCfg


class Trainer:
    """single gpu training"""

    def __init__(self, cfg: TrainerCfg) -> None:
        self.cfg = cfg
        self.device = cfg.device

    def loop(
        self,
        lr_scheduler,
        optimizer,
        train_datagen,
        eval_datagen,
        model,
    ):
        device = self.device
        model.to(device)

        train_size = len(train_datagen)
        eval_size = len(eval_datagen)

        def fwd(self, batch):
            audio_mels = batch["audio_mels"]
            cap_ids = batch["cap_ids"]
            cap_ids_attention_mask = batch["cap_attention_mask"]
            prompt_ids = batch["prompt_ids"]
            prompt_ids_attention_mask = batch["prompt_attention_mask"]
            end_prompt_ids = batch["end_prompt_ids"]
            end_prompt_ids_attention_mask = batch["end_prompt_attention_mask"]

            audio_embeds = self.audio_encoder(audio_mels)
            # print('audio_embeds', audio_embeds.mean(dim=1), audio_embeds.std(dim=1))
            bs, audio_seq = audio_embeds.shape[:2]

            attention_mask = torch.concat(
                (
                    prompt_ids_attention_mask,
                    torch.ones(bs, audio_seq).to(cap_ids.device),
                    end_prompt_ids_attention_mask,
                    cap_ids_attention_mask,
                ),
                dim=1,
            )

            cap_embeds = self.llm.model.embed_tokens(cap_ids)
            prompt_embeds = self.llm.model.embed_tokens(prompt_ids)
            end_prompt_embeds = self.llm.model.embed_tokens(end_prompt_ids)
            inputs_embeds = torch.concat(
                (
                    prompt_embeds,
                    audio_embeds.to(cap_embeds.dtype),
                    end_prompt_embeds,
                    cap_embeds,
                ),
                dim=1,
            )

            # print('cap_embeds', cap_embeds.mean(dim=1), cap_embeds.std(dim=1))
            mout = self.llm(
                inputs_embeds=inputs_embeds,
                # output_attentions=True,
                # output_hidden_states=True,
                attention_mask=attention_mask,
                # use_cache=False,
            )

            return mout, audio_embeds.shape[1]

        for ep in range(self.cfg.epoch):
            mean_train_losses = []
            mean_eval_losses = []

            ts_fwd = []
            ts_backprop = []

            # <train>
            train_losses = []
            for ti, local_batch in enumerate(train_datagen):
                # Transfer to GPU
                batch = {
                    k: v.to(device)
                    for k, v in local_batch.items()
                    if not k.startswith("_")
                }

                # compute
                with timing_context(ts_fwd):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        mout, audio_seq = fwd(model, batch)

                # calculate loss
                prompt_ids_seq = local_batch["prompt_ids"].shape[1]
                end_prompt_ids_seq = local_batch["end_prompt_ids"].shape[1]
                logits_start = prompt_ids_seq + audio_seq + end_prompt_ids_seq

                # remove the last output
                logits = mout.logits
                # remove the prompt and audio seq from logits
                # calculation; additionally, remove the final item
                logits = logits[:, logits_start:-1, :].contiguous()

                # calculate target using only `cap_ids`
                targets = batch["cap_ids"][:]
                targets = targets[:, 1:]

                # print("logits", logits.view(-1, logits.shape[-1]).mean(dim=1), logits.view(-1, logits.shape[-1]).std(dim=1))

                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]), targets.view(-1)
                )

                with timing_context(ts_backprop):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                train_losses.append(loss.detach().cpu())

                batch_avg_time_sec = np.mean(ts_backprop[-100:]) + np.mean(
                    ts_fwd[-100:]
                )

                # <house keeping>
                ts_backprop = ts_backprop[-100:]
                ts_fwd = ts_fwd[-100:]

                print_status(
                    mode="tr",
                    remaining_time_sec=(train_size - (ti + 1)) * batch_avg_time_sec,
                    running_loss=train_losses,
                )
                # </house keeping>
            # </train>

            # <eval>
            eval_losses = []
            with torch.no_grad():
                for local_batch in eval_datagen:
                    # Transfer to GPU
                    batch = {
                        k: v.to(device)
                        for k, v in local_batch.items()
                        if not k.startswith("_")
                    }

                    # compute
                    with timing_context(ts_fwd):
                        with torch.amp.autocast(
                            device_type="cuda", dtype=torch.float16
                        ):
                            mout, audio_seq = fwd(model, batch)

                    prompt_ids_seq = local_batch["prompt_ids"].shape[1]
                    end_prompt_ids_seq = local_batch["end_prompt_ids"].shape[1]
                    logits_start = prompt_ids_seq + audio_seq + end_prompt_ids_seq

                    logits = mout.logits
                    # remove the prompt and audio seq from logits
                    # calculation; additionally, remove the final item
                    logits = logits[:, logits_start:-1, :].contiguous()

                    # calculate target using only `cap_ids`
                    targets = batch["cap_ids"][:]
                    targets = targets[:, 1:]

                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]), targets.view(-1)
                    )

                    eval_losses.append(loss.detach().cpu())

                    print_status(
                        mode="ev",
                        remaining_time_sec=0,
                        running_loss=eval_losses,
                    )
            # </eval>

            # Adjust the learning rate
            if lr_scheduler:
                lr_scheduler.step()

            mean_train_losses.append(np.mean(train_losses[-1000:]))
            mean_eval_losses.append(np.mean(eval_losses[-1000:]))

            print(
                f"{ep}:tloss{mean_train_losses[-1]:.4f},eloss{mean_eval_losses[-1]:.4f}"
            )

            if self.cfg.model_out_dir:
                if ep != 0 and (ep % self.cfg.model_save_freq == 0):
                    eval_loss_4f = f"{mean_eval_losses[-1]:.4f}"
                    model.save(
                        SaveCfg(
                            epoch=ep,
                            out_dir=self.cfg.model_out_dir,
                            eval_loss_4f=eval_loss_4f,
                        ),
                    )


def status_update_line(status):
    return "\x1b[2K%s" % status


def print_status(*, mode, remaining_time_sec, running_loss):
    print(
        status_update_line(
            "[{}] eta={} loss={:.4f}".format(
                mode,
                seconds_to_human_readable(remaining_time_sec),
                np.mean(running_loss[-100:]),
                # extra_info,
            )
        ),
        end="\r",
    )


def seconds_to_human_readable(elapsed):
    # Calculate days, hours, minutes, and seconds
    days, remainder = divmod(elapsed, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)

    # Format the result as a string
    result = ""
    if days > 0:
        result += f"{int(days)}d"

    if hours > 0:
        if result:
            result += ", "
        result += f"{int(hours)}h"

    # If no days or hours, show minutes
    if not result:
        minutes, seconds = divmod(remainder, 60)
        result += f"{int(minutes)}m"

    return result


@contextmanager
def timing_context(sample):
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    sample.append(elapsed_time)
