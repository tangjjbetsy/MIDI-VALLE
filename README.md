# MIDI-VALLE: Piano Performance Synthesis with VALL-E
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J16U55C-uBYgMDasUC7-Ku8zirzjFVQb?usp=sharing)
[![arXiv](https://img.shields.io/badge/arXiv-2501.10222v1-b31b1b.svg)](https://arxiv.org/abs/2501.10222v1)
![Conference](https://img.shields.io/badge/Conference-ISMIR%202025-blue)

This repository contains the official implementation of our ISMIR 2025 [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10890623)

**"MIDI-VALLE: Improving Expressive Piano Performance Synthesis Through Neural Codec Language Modelling"**

by Jingjing Tang, Xin Wang, Zhe Zhang, Junichi Yamagishi, Geraint Wiggins, and György Fazekas.

## Overview
This repository provides an implementation of MIDI-VALLE, a system that adapts the [VALL-E]((https://arxiv.org/abs/2301.02111)) model for expressive piano performance synthesis. The system is designed to synthesize expressive piano performances from MIDI files. We adapt the [unofficial VALL-E implementation](https://github.com/lifeiteng/vall-e) and modify it to handle MIDI data.

## Dataset & Checkpoints
Due to the copyright issues, we cannot provide the audios for the ATEPP dataset, but the midi files are available in the [ATEPP repository](https://github.com/tangjjbetsy/ATEPP). The checkpoints for Piano-Enodec and the MIDI-VALLE could be downloaded from [Zenodo](). 

## Installation
To install the necessary dependencies, run the following command:
```
# AudioCraft
pip install audiocraft==1.3.0

# PyTorch
pip install torch==2.1.0 torchaudio==2.1.0
pip install torchmetrics==0.11.1

# fbank
pip install librosa==0.11.0

# lhotse
pip uninstall lhotse
pip install git+https://github.com/lhotse-speech/lhotse

# k2
# find the right version in https://huggingface.co/csukuangfj/k2
pip install https://huggingface.co/csukuangfj/k2/resolve/main/cpu/k2-1.24.4.dev20240223+cpu.torch2.1.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# icefall
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=`pwd`/../icefall:$PYTHONPATH
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.zshrc
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.bashrc
cd -
source ~/.zshrc

# midi
pip install pretty_midi==0.2.10
pip install miditok==3.0.2

# midi-valle
cd midi-valle
pip install -e .
```

## Prepare Dataset
The midi files and audio files were cut into segments first and then were tokenized.

```
cd egs/atepp
PATH_TO_ATEPP_CORPUS="ATEPP"
PATH_TO_OUTPUT_DIR="ATEPP-midi-valle"
PATH_TO_MIDI_DIR="ATEPP-midi-valle/midi_seg"

# Prepare the midi tokens
cd local
python data_prepare.py --mode prepare_segments --data-path ${PATH_TO_ATEPP_CORPUS} --out-path ${PATH_TO_OUTPUT_DIR} --audio
# Tokenise the midi files
python midi_tokenize.py --data_folder ${PATH_TO_MIDI_DIR} --target_folder ${PATH_TO_MIDI_DIR}
# Prepare the audio tokens
python atepp.py
bash prepare.sh --stage -2 --stop-stage 3
```

## Training
```
# path to save the model checkpoints
exp_dir=exp/midi-valle 

# Train the model
python3 bin/trainer.py --num-buckets 12 --save-every-n 20000 --valid-interval 10000 \
    --model-name valle --share-embedding true --norm-first true --add-prenet false   \
    --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1  \
    --base-lr 0.05 --warmup-steps 20 --average-period 0 \
    --num-epochs 20 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 8    \
    --exp-dir ${exp_dir} --dataset atepp --max-duration 200 --keep-last-k 3 \
    --train-stage 0 --num-quantizers 4 --manifest-dir "data_large_new/tokenized" --filter-min-duration 3 \
    --inf-check True  --world-size 1
```

## Inference
```
# Inference on a single midi file (prompts are required)
PROMPT_MIDI="ATEPP-midi-valle/prompt_midi.midi"
PROMPT_WAV="ATEPP-midi-valle/prompt_wav.wav"
cd local
# Segment the midi file
python data_prepare.py --mode prepare_segments --data-path ${PATH_TO_ATEPP_CORPUS} --out-path ${PATH_TO_OUTPUT_DIR}
# Prepare the prompt midi and audio files
python data_prepare.py --mode process_for_performance_inference --data-path ${PATH_TO_OUTPUT_DIR} --out-path ${PATH_TO_OUTPUT_DIR} \
    --prompt-midi-dir ${PROMPT_MIDI} --prompt-wav-dir ${PROMPT_WAV}
# Tokenize the midi files
python midi_tokenize.py --data_folder ${PATH_TO_OUTPUT_DIR} --target_folder ${PATH_TO_OUTPUT_DIR}

# For any midi segment, you can run the inference as follows:
midi_cat="ATEPP-midi-valle/midi_seg/segment_0001_cat.npy"
output_file="segment_0001"

# Inference
output_dir=exp/midi-valle/output
checkpoint=exp/midi-valle/best-valid-loss.pt
python3 bin/infer.py \
  --output-dir ${output_dir} \
  --checkpoint=${checkpoint} \
  --midi-prompts ${PROMPT_MIDI} \
  --audio-prompts ${PROMPT_WAV} \
  --midi ${midi_cat} \
  --audio-tokenizer Encodec32FT \
  --output-file ${output_file}
```

## Demo
You can listen to the demo samples on our [project page](https://tangjjbetsy.github.io/MIDI-VALLE/).

## Contact
Jingjing Tang: `jingjing.tang@qmul.ac.uk`

## License
The code is licensed under Apache License Version 2.0, following the [unofficial implementation](https://github.com/lifeiteng/vall-e) of VALLE. The pretrained model is licensed under the Creative Commons License: Attribution 4.0 International http://creativecommons.org/licenses/by/4.0/legalcode

## Acknowledgements
This work was supported by the UKRI Centre for Doctoral Training in Artificial Intelligence and Music [grant number EP/S022694/1] and the National Institute of Informatics, Japan. J. Tang is a research student jointly funded by the China Scholarship Council [grant number 202008440382] and Queen Mary University of London. G. Wiggins received funding from the Flemish Government under the "Onderzoeksprogramma Artificiële Intelligentie (AI) Vlaanderen". We thank the reviewers for their valuable feedback, which helped improve the quality of this work.