# MIDI-VALLE: Piano Performance Synthesis with VALL-E
[![Codes](https://img.shields.io/badge/GitHub-Midi--VALLE-blue?logo=github)](https://github.com/tangjjbetsy/MIDI-VALLE)
[![Pre-trained Models](https://img.shields.io/badge/Models-Zenodo-9cf?logo=zenodo)](https://zenodo.org/records/15976272)
[![arXiv](https://img.shields.io/badge/arXiv-2507.08530-b31b1b.svg)](https://arxiv.org/abs/2507.08530)
[![Conference](https://img.shields.io/badge/Conference-ISMIR%202025-green)](https://ismir2025.ismir.net/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](https://zenodo.org/records/15976272/files/LICENSE?download=1)


This repository contains the official implementation of our ISMIR 2025 [paper](https://arxiv.org/abs/2507.08530)

**"MIDI-VALLE: Improving Expressive Piano Performance Synthesis Through Neural Codec Language Modelling"**

by Jingjing Tang, Xin Wang, Zhe Zhang, Junichi Yamagishi, Geraint Wiggins, and György Fazekas.

## Overview
This repository provides an implementation of MIDI-VALLE, a system that adapts the [VALL-E]((https://arxiv.org/abs/2301.02111)) model for expressive piano performance synthesis. The system is designed to synthesize expressive piano performances from MIDI files. We adapt the [unofficial VALL-E implementation](https://github.com/lifeiteng/vall-e) and modify it to handle MIDI data. It's recommended to use the [MIDI-VALLE Colab](https://colab.research.google.com/drive/1JuQ7uv8lPbdQhF7xCrcGFg-rCFQ0tPgU?usp=sharing) for a quick attempt to synthesize expressive piano performances.

## Dataset & Checkpoints
Due to the copyright issues, we cannot provide the audios for the ATEPP dataset, but the midi files are available in the [ATEPP repository](https://github.com/tangjjbetsy/ATEPP). The checkpoints for Piano-Enodec and the MIDI-VALLE could be downloaded from [Zenodo](https://zenodo.org/records/15976272). Please put the checkpoints in the `midi-valle/egs/atepp/checkpoints` directory.

## How to Train and Infer
### Environment Setup
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

# others
pip install pretty_midi==0.2.10
pip install miditok==3.0.2
pip install numpy==1.26.4
pip install transformers==4.41.0 
pip install soundfile
pip install scipy==1.11.4

# midi-valle
cd midi-valle
pip install -e .
```

### Prepare Dataset
The midi files and audio files were cut into segments first and then were tokenized.

```
cd egs/atepp
PATH_TO_ATEPP_CORPUS="ATEPP"
PATH_TO_OUTPUT_DIR="ATEPP-midi-valle"
PATH_TO_MIDI_DIR="ATEPP-midi-valle/midi_seg"

# Prepare the midi tokens
cd local
python data_prepare.py prepare_segments --data_path ${PATH_TO_ATEPP_CORPUS} --out_path ${PATH_TO_OUTPUT_DIR} --audio
# Tokenise the midi files
python midi_tokenize.py --data_folder ${PATH_TO_OUTPUT_DIR} --target_folder midi_seg
# Prepare the audio tokens
python atepp.py
bash prepare.sh --stage -2 --stop-stage 3
```

### Training
To train the MIDI-VALLE model, you can use the following command. Make sure to adjust the parameters according to your needs. The training will take a long time, so we recommend using a GPU with at least 24GB of memory.

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
    --train-stage 0 --num-quantizers 4 --manifest-dir "data/tokenized" --filter-min-duration 3 \
    --inf-check True  --world-size 1
```

### Inference
We recommend using the [MIDI-VALLE Colab](https://colab.research.google.com/drive/1JuQ7uv8lPbdQhF7xCrcGFg-rCFQ0tPgU?usp=sharing) for a quick attempt to synthesize expressive piano performances. If you want to run the inference locally, please follow the steps below.

```
##### Prepare the data #####
cd egs/atepp
# Inference on a single midi file (prompts are required)
PROMPT_MIDI="prompts/prompt_A.midi"
PROMPT_WAV="prompts/prompt_A.wav"
PATH_TO_DATA="path_to_your_midi_folder"
PATH_TO_OUTPUT_DIR="path_to_your_output_folder" # For saving intermediate files

cd local
# Segment the target midi file
python data_prepare.py prepare_segments --data_path ${PATH_TO_DATA} --out_path ${PATH_TO_OUTPUT_DIR}

# Integrate the prompt midi and the target midi
python data_prepare.py process_for_performance_inference_colab --data_path ${PATH_TO_OUTPUT_DIR} --out_path ${PATH_TO_OUTPUT_DIR} --prompt-midi-dir ${PROMPT_MIDI} --prompt-wav-dir ${PROMPT_WAV}

# Tokenize the midi files
python midi_tokenize.py --data_folder ${PATH_TO_OUTPUT_DIR} --target_folder cat_target 
# cat_XXX could be the name of the target midi file, check the sub-folder under ${PATH_TO_OUTPUT_DIR}
```

After running the above commands, you will have segmented midi files concatenated with the prompt midi and tokenised.

```
# Inference on the target midi files
cd egs/atepp # Change to the directory where the inference script is located

midi="target_midi_0_cat.npy|target_midi_1_cat.npy|..." # The path to midi segments to be synthesized, separated by '|'
output_file="output"
output_dir="midi-valle/egs/atepp/output"
checkpoint="midi-valle/egs/atepp/checkpoints/best-valid-loss.pt"

python3 bin/infer.py \
  --output-dir ${output_dir} \
  --checkpoint=${checkpoint} \
  --midi-prompts ${PROMPT_MIDI} \
  --audio-prompts ${PROMPT_WAV} \
  --midi ${midi} \
  --audio-tokenizer Encodec32FT \
  --output-file ${output_file}
```

### Objective Metrics
The Chroma and spectrogram distance metrics can be computed using the tools release in [here](https://github.com/nii-yamagishilab/score-to-audio/tree/main/objective_eval).

For the FAD, please follow the instructions in the `fadtk` (copied from [microsoft released version](https://github.com/microsoft/fadtk) and added Piano-Encodec as a custom model) and run the following command:

```
fadtk encodec-music /path/to/baseline/audio /path/to/evaluation/audio
```

The performances used for the evaluation are listed in the:
```
fadtk/datasets/ATEPP-1.2-clean-sample.csv,
fadtk/datasets/maestro-v3.0.0-sample.csv, 
fadtk/datasets/pijama_sample.csv
```

## Demo
You can listen to the demo samples on our [project page](https://tangjjbetsy.github.io/MIDI-VALLE/).

## Contact
Jingjing Tang: `jingjing.tang@qmul.ac.uk`

## License
The code is licensed under Apache License Version 2.0, following the [unofficial implementation](https://github.com/lifeiteng/vall-e) of VALLE. The pretrained model is licensed under the Creative Commons License: Attribution 4.0 International http://creativecommons.org/licenses/by/4.0/legalcode

## Acknowledgements
This work was supported by the UKRI Centre for Doctoral Training in Artificial Intelligence and Music [grant number EP/S022694/1] and the National Institute of Informatics (NII), Japan. J. Tang is a research student jointly funded by the China Scholarship Council [grant number 202008440382] and Queen Mary University of London. G. Wiggins received funding from the Flemish Government under the "Onderzoeksprogramma Artificiële Intelligentie (AI) Vlaanderen". We thank the reviewers for their valuable feedback, which helped improve the quality of this work. 
