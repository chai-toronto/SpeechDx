import librosa
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
from chatterbox.models.s3gen import S3GEN_SR # 24000
from chatterbox.models.s3tokenizer import S3_SR # 16000
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals


class Chatterbox(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.model = ChatterboxTurboTTS.from_pretrained(device=device)
        self.output_hidden_states = False

    def forward(self, wav, lengths = None):
        """ One waveform at a time. """
        with torch.no_grad():
            conditions = self.prepare_conditionals(wav)
            emb = self.model.t3.prepare_conditioning(conditions.t3)
        return emb.detach().clone()

    def prepare_conditionals(self, s3gen_ref_wav, exaggeration=0.5, norm_loudness=True):
        ## Load and norm reference wav
        # s3gen_ref_wav, _sr = torchaudio.load(wav, sr=S3GEN_SR)
        #
        # assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        if s3gen_ref_wav.ndim == 2:
            s3gen_ref_wav = s3gen_ref_wav.mean(axis=0)
            s3gen_ref_wav = s3gen_ref_wav.numpy()

        if norm_loudness:
            s3gen_ref_wav = self.model.norm_loudness(s3gen_ref_wav, S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.model.DEC_COND_LEN]
        s3gen_ref_dict = self.model.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.model.device)

        # Speech cond prompt tokens
        if plen := self.model.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.model.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.model.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.model.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.model.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.model.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.model.device)
        return Conditionals(t3_cond, s3gen_ref_dict)

