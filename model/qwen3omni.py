import torch
import torch.nn as nn

from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

USE_AUDIO_IN_VIDEO = False

class Qwen3Omni(nn.Module):
    def __init__(self, source, freeze_encoder=True, sample_rate=16000, output_hidden_states=False):
        super().__init__()
        assert not output_hidden_states, "Not implemented yet"
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                                                                source,
                                                                dtype="auto",
                                                                )
        self.model.disable_talker()

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(source)

        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        self.freeze_encoder = freeze_encoder
        self.sample_rate = sample_rate
        self.output_hidden_states = output_hidden_states

    def _get_device_dtype(self):
        p = next(self.model.parameters())
        return p.device, p.dtype

    @staticmethod
    def get_conversation(audio):
        """1D input"""
        prompt = ("Detect any respiratory symptom in the following audio speech. Feel free to deliberate with yourself. "
                  "Afterwards, begin your answer with \"FINAL ANSWER:\", followed by 1 if there is a symptom and 0 otherwise.")
        return [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio},
                        {"type": "text", "text": prompt},
                    ]
                }
            ]


    def forward(self, x, lengths=None):
        if isinstance(x, torch.Tensor):
            x = x.float().cpu().numpy()

        prompts = [self.get_conversation(audio) for audio in x]
        text = self.processor.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)
        audios, _, _ = process_mm_info(prompts, use_audio_in_video=USE_AUDIO_IN_VIDEO)

        device, dtype = self._get_device_dtype()

        with torch.amp.autocast(device.type, enabled=False):
            inputs = self.processor(text=text,
                                   audio=audios,
                                   return_tensors="pt",
                                   padding=True,
                                   use_audio_in_video=USE_AUDIO_IN_VIDEO)

        inputs = inputs.to(device).to(dtype)

        text_ids, _ = self.model.generate(**inputs,
                                              return_audio=False,
                                              thinker_return_dict_in_generate=True,
                                              use_audio_in_video=USE_AUDIO_IN_VIDEO)

        texts = self.processor.batch_decode(text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)

        final_ans_idx = [text.find("FINAL ANSWER:") for text in texts]

        final_ans_int = []

        for i, idx in enumerate(final_ans_idx):
            ans = texts[i][idx:]
            ans = 1 if "1" in ans else 0
            final_ans_int.append(ans)

        return torch.tensor(final_ans_int, dtype=torch.float).unsqueeze(1)


if __name__ == "__main__":
    import soundfile as sf

    paths = ["/Users/lkieu/PycharmProjects/Audio-Health-Benchmark/data/c9s_t1/processed/audio/0A0ULgZntg/2020-11-20-17_40_23_289245/voice_0A0ULgZntg_1605894017115.wav",
             "/Users/lkieu/PycharmProjects/Audio-Health-Benchmark/data/c9s_t1/processed/audio/0AP3oRQmiDQV/2020-11-15-19_07_30_223783/audio_file_read.wav"]

    model = Qwen3Omni(source="Qwen/Qwen3-Omni-30B-A3B-Instruct", freeze_encoder=True)

    wavs = []
    for p in paths:
        wav, sr = sf.read(p, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        wavs.append(wav)

    with torch.no_grad():
        outputs = model(wavs)

    for p, ans in zip(paths, outputs):
        print(f"{p}\n  -> symptom={ans.item()}")
