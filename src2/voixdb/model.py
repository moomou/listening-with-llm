import logging
import pathlib as pl

import torch
import torch.nn as nn
import whisper
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer,
)

NVME2 = pl.Path("/media/2nvme")

MODEL_DIR = NVME2 / "models" / "whisper-large-v3"
WHISPER_MODEL_FILE = MODEL_DIR / "large-v3.pt"
WHISPER_AUDIO_BIN = "/media/2nvme/proc/voixdb/frozen/audio_encoder.statedict"

ORCA = "/media/2nvme/llm/Mistral-7B-OpenOrca"
# HERMES = "/media/2nvme/llm/OpenHermes-2.5-Mistral-7B"
# Loads and works with bnb but outputs gibberish
# QWEN = "/media/2nvme/llm/Qwen-7B-Chat"

MODEL = ORCA

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, audio_encoder, llm):
        super().__init__()

        self.audio_encoder = audio_encoder
        self.llm = llm

        for p in self.llm.parameters():
            p.requires_grad = False

    def save(self, save_cfg):
        out = save_cfg.output_filename()
        logger.info("saving out to", out)

        torch.save(
            self.audio_encoder.proj.state_dict(),
            out,
        )

    def load(self, save_cfg):
        out = save_cfg.state_from_epoch()
        logger.info("loading from out to", out)

        self.audio_encoder.proj.load_state_dict(torch.load(out))

    def forward(self, batch):
        audio_mels = batch["audio_mels"]
        cap_ids = batch["cap_ids"]
        cap_ids_attention_mask = batch["cap_attention_mask"]
        prompt_ids = batch["prompt_ids"]
        prompt_ids_attention_mask = batch["prompt_attention_mask"]
        end_prompt_ids = batch["end_prompt_ids"]
        end_prompt_ids_attention_mask = batch["end_prompt_attention_mask"]

        audio_embeds = self.audio_encoder(audio_mels)
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

        mout = self.llm(
            inputs_embeds=inputs_embeds,
            # output_attentions=True,
            # output_hidden_states=True,
            attention_mask=attention_mask,
            # use_cache=False,
        )

        return mout, audio_embeds.shape[1]


class TunableWhisperAudioEncoder(nn.Module):
    def __init__(self, *, audio_encoder=None, output_embedding_size=4096):
        """
        args
            output_embedding_size: int = 4096 / mistral default embedding size
        """
        super().__init__()

        if audio_encoder is None:
            audio_encoder = load_whisper_v3_audio_encoder()

        self.audio_encoder = audio_encoder
        self.proj = TrainableSubmodule(output_embedding_size=output_embedding_size)

        # # Freeze all parameters
        # # TODO apply LoRA on this encoder
        for param in audio_encoder.parameters():
            param.requires_grad = False

    def forward(self, mels):
        res = self.audio_encoder(mels)
        res = self.proj(res)
        return res


class TrainableSubmodule(nn.Module):
    def __init__(self, output_embedding_size):
        super().__init__()

        # TODO: init from BERT
        # create a trainable proj layer
        # self.cnn1 = nn.Conv1d(1280, 640, 3, stride=2, dilation=2, bias=False)
        # self.cnn2 = nn.Conv1d(640, 1280, 3, stride=2, dilation=2, bias=False)
        # self.cnn3 = nn.Conv1d(640, 1280, 3, stride=2, dilation=2, bias=False)
        self.pool = nn.AdaptiveAvgPool1d(250)
        self.proj = nn.Linear(1280, output_embedding_size, bias=False)
        self.ln1 = nn.LayerNorm(1280)
        # self.ln2 = nn.LayerNorm(640)
        # self.ln2 = nn.LayerNorm(1280)

        # for layer in [self.cnn1, self.cnn2, self.proj, self.ln1, self.ln2]:
        # nn.init.kaiming_uniform_(layer.weight, mode="fan_in", nonlinearity="linear")
        # nn.init.constant_(layer.weight, 0)
        #     # Optionally, you can initialize the biases as well
        #     if layer.bias is not None:
        #         nn.init.constant_(layer.bias, 0)

    def forward(self, audio_embeds):
        # res = self.cnn1(audio_embeds.transpose(-2, -1))
        # res = self.ln1(res.transpose(-2, -1))

        # res = self.cnn2(res.transpose(-2, -1))
        # res = self.ln2(res.transpose(-2, -1))

        # res = self.cnn3(res.transpose(-2, -1))
        # res = self.ln3(res.transpose(-2, -1))
        res = audio_embeds
        res = self.pool(res.transpose(-2, -1))
        res = self.proj(self.ln1(res.transpose(-2, -1)))
        return res


def load_whisper_v3_audio_encoder(
    *,
    n_mels=128,
    n_audio_ctx=1500,
    n_audio_state=1280,
    n_audio_head=20,
    n_audio_layer=32,
):
    m = whisper.model.AudioEncoder(
        n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer
    )
    m.load_state_dict(torch.load(WHISPER_AUDIO_BIN))
    return m


def load_llm():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_compute_type=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        # bnb_4bit_use_double_quant=True,
        # load_in_8bit=True,
    )

    # Load model
    tokenizer = LlamaTokenizer.from_pretrained(
        MODEL,
        trust_remote_code=False,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        trust_remote_code=False,
        use_safetensors=True,
        quantization_config=bnb_config,
    )

    return tokenizer, model
