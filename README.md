# MuFun
<div align="center">
    <a href="https://arxiv.org/abs/2508.01178"><img src="https://img.shields.io/badge/arXiv-2508.01178-b31b1b" alt="version"></a>
    <a href="https://huggingface.co/collections/Yi3852/mufun-68943d4ad905f4e23e35b86d"><img src="https://img.shields.io/badge/HuggingFace-Collections-ffc107" alt="version"></a>
</div>

training and fine-tuning code for the MuFun model proposed in [Advancing the Foundation Model for Music Understanding](https://arxiv.org/abs/2508.01178)

detailed instructions and examples will be added soon

Our main training code is adapted from [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) to support audio input, as for reinforcement learning we modify the HuggingFace TRL library.

## Inference Code

for inference it's not necessary to install this repo, only some audio processing packages like mutagen, torchaudio are needed

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
hf_path = 'Yi3852/MuFun-Instruct' # or 'Yi3852/MuFun-Base'
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False)
device='cuda'
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True, torch_dtype="bfloat16")
model.to(device)

# single audio
# during inference the audio(converted to a sequence of embeddings) will be placed in the position of <audio> tag in the prompt
aud="/path/to/your/song.mp3"
inp="\n<audio>Can you listen to this song and tell me its lyrics?" 
res=model.chat(prompt=inp, audio_files=aud, tokenizer=tokenizer)
print(res)

# multiple audios
# for multiple songs each will be placed in the coresponding <audio> tag in the prompt
aud=["/path/to/your/song1.mp3", '/path/to/your/song2.mp3']
inp="\n<audio> This is song1. <audio> This is song2. Which song do you like more? Tell me the reason."
res=model.chat(prompt=inp, audio_files=aud, tokenizer=tokenizer)
print(res)

# analyze only a specific segment of audio using the segs parameter
# format is [start_time, end_time](in seconds), for multiple audios segs can be passed like [[0,30],[60,90]], [None,[0,30.0]]
aud="/path/to/your/song.mp3"
inp="\n<audio>How is the rhythm of this music clip?"
res=model.chat(prompt=inp, audio_files=aud, segs=[0,30.0], tokenizer=tokenizer)
print(res)

# set audio_files=None will work, however it is not recommended to use it as a text model
```

## Citation

```bibtex
@misc{jiang2025advancingfoundationmodelmusic,
      title={Advancing the Foundation Model for Music Understanding}, 
      author={Yi Jiang and Wei Wang and Xianwen Guo and Huiyun Liu and Hanrui Wang and Youri Xu and Haoqi Gu and Zhongqian Xie and Chuanjiang Luo},
      year={2025},
      eprint={2508.01178},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.01178}, 
}
