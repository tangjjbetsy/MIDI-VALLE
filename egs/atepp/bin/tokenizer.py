#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize Text and EnCodec Audio.

Usage example:
    python3 bin/tokenizer.py \
        --src_dir ./data/manifests --output_dir ./data/tokenized

"""
import argparse
import logging
import os
from pathlib import Path
import numpy as np

import torch
import torch.multiprocessing
from icefall.utils import get_executor
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm

from valle.data import (
    AudioTokenConfig,
    AudioTokenConfig32,
    AudioTokenExtractor,
    AudioTokenExtractor32,
    AudioTokenExtractor32FT,
    # TextTokenizer,
    # tokenize_text,
)
from valle.data.fbank import get_fbank_extractor
from valle.utils import SymbolTable

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Path to the manifest files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Path to the tokenized files",
    )
    
    parser.add_argument(
        "--audio-extractor",
        type=str,
        default="Encodec",
        help="Encodec or Fbank",
    )
    parser.add_argument(
        "--dataset-parts",
        type=str,
        default="test train validation",
        help="Space separated dataset parts",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="atepp",
        help="prefix of the manifest file",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="jsonl.gz",
        help="suffix of the manifest file",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=100.0,
        help="The maximum number of audio seconds in a batch."
        "Determines batch size dynamically.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    dataset_parts = args.dataset_parts.replace("--dataset-parts", "").strip()
    if dataset_parts == "all":  # LibriTTS
        dataset_parts = [
            "test",
            "train",
            "validation"
        ]
    else:
        dataset_parts = dataset_parts.replace("-p", "").strip().split(" ")

    assert len(dataset_parts) >= 1

    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=args.src_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        types=["recordings", "supervisions", "cuts"],
    )

    audio_extractor = None
    if args.audio_extractor:
        if args.audio_extractor == "Encodec":
            audio_extractor = AudioTokenExtractor(AudioTokenConfig())
        elif args.audio_extractor == "Encodec32":
            audio_extractor = AudioTokenExtractor32(AudioTokenConfig32())
        elif args.audio_extractor == "Encodec32FT":
            audio_extractor = AudioTokenExtractor32FT(AudioTokenConfig32())
        else:
            assert args.audio_extractor == "Fbank"
            audio_extractor = get_fbank_extractor()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    num_jobs = min(4, os.cpu_count())
    logging.info(f"dataset_parts: {dataset_parts} manifests {len(manifests)}")

    prefix = args.prefix
    if prefix and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    with get_executor() as ex:
        for partition, m in manifests.items():
            logging.info(
                f"Processing partition: {partition} CUDA: {torch.cuda.is_available()}"
            )
            try:
                cut_set = CutSet.from_manifests(
                    recordings=m["recordings"],
                    supervisions=m["supervisions"],
                )
            except Exception:
                cut_set = m["cuts"]
            
            # AudioTokenizer
            if args.audio_extractor:
                if args.audio_extractor == "Encodec":
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_encodec_{partition}"
                    )
                elif args.audio_extractor == "Encodec32":
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_encodec_{partition}"
                    )
                elif args.audio_extractor == "Encodec32FT":
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_encodec_{partition}"
                    )
                else:
                    storage_path = (
                        f"{args.output_dir}/{args.prefix}_fbank_{partition}"
                    )

                # if args.prefix.lower() in ["atepp"]:
                #     cut_set = cut_set.resample(48000)
                    # https://github.com/lifeiteng/vall-e/issues/90
                    # if args.prefix == "aishell":
                    #     # NOTE: the loudness of aishell audio files is around -33
                    #     # The best way is datamodule --on-the-fly-feats --enable-audio-aug
                    #     cut_set = cut_set.normalize_loudness(
                    #         target=-20.0, affix_id=True
                    #     )

                with torch.no_grad():
                    if (
                        torch.cuda.is_available()
                        and args.audio_extractor in ["Encodec", "Encodec32", "Encodec32FT"]
                    ):
                        cut_set = cut_set.compute_and_store_features_batch(
                            extractor=audio_extractor,
                            storage_path=storage_path,
                            num_workers=num_jobs,
                            batch_duration=args.batch_duration,
                            collate=False,
                            overwrite=True,
                            storage_type=NumpyHdf5Writer,
                        )
                    else:
                        cut_set = cut_set.compute_and_store_features(
                            extractor=audio_extractor,
                            storage_path=storage_path,
                            num_jobs=num_jobs if ex is None else 64,
                            executor=ex,
                            storage_type=NumpyHdf5Writer,
                        )

            # MidiTokens
            for c in tqdm(cut_set):
                tokens = np.load(c.supervisions[0].text).tolist()
                c.supervisions[0].custom = {}
                c.supervisions[0].custom["tokens"] = {"midi": tokens}

            cuts_filename = f"{prefix}cuts_{partition}.{args.suffix}"
            cut_set.to_file(f"{args.output_dir}/{cuts_filename}")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
