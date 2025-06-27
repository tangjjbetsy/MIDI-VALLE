"""
This file was developed as the lhost reciept for preparing the data.
"""

import logging
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike

def prepare_atepp(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None, tokenizer: str = "expression" \
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    transcript_dir = corpus_dir / "midi_seg"
    assert transcript_dir.is_dir(), f"No such directory: {transcript_dir}"
    if tokenizer == "expression":
        transcript_files = transcript_dir.rglob("**/*.npy")
        transcript_files = [file for file in transcript_files if 'remi' not in str(file)]
    else:
        transcript_files = transcript_dir.rglob("**/*_remi.npy")
        
    transcript_dict = {}
    for file in transcript_files:
        if tokenizer == "expression":
            transcript_dict[file.stem] = str(file)
        else:
            transcript_dict[file.stem.split("_remi")[0]] = str(file)
    
    manifests = defaultdict(dict)
    dataset_parts = ["test", "validation", "train"]
    for part in tqdm(
        dataset_parts,
        desc="Process atepp audio and text, it takes about 160 seconds.",
    ):
        logging.info(f"Processing atepp subset: {part}")
        # Generate a mapping: utt_id -> (audio_path, audio_info, speaker, text)
        recordings = []
        supervisions = []
        wav_path = corpus_dir / "wav_seg" / f"{part}"
        for audio_path in tqdm(list(wav_path.rglob("**/*.wav"))):
            idx = audio_path.stem
            speaker = audio_path.stem.split("_")[0]
            language = audio_path.stem.split("_")[1]
            if idx not in transcript_dict:
                logging.warning(f"No transcript: {idx}")
                logging.warning(f"{audio_path} has no transcript.")
                continue
            text = transcript_dict[idx]
            if not audio_path.is_file():
                logging.warning(f"No such file: {audio_path}")
                continue
            recording = Recording.from_file(audio_path)
            if recording.duration <= 0:
                print(audio_path)
                continue
            recordings.append(recording)
            segment = SupervisionSegment(
                id=idx,
                recording_id=idx,
                start=0.0,
                duration=recording.duration,
                channel=0,
                speaker=speaker,
                language=language,
                text=text.strip(),
            )
            supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
        validate_recordings_and_supervisions(recording_set, supervision_set)

        if output_dir is not None:
            supervision_set.to_file(
                output_dir / f"atepp_supervisions_{part}.jsonl.gz"
            )
            recording_set.to_file(output_dir / f"atepp_recordings_{part}.jsonl.gz")

        manifests[part] = {"recordings": recording_set, "supervisions": supervision_set}

    return manifests

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the ATEPP dataset manifests."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Pathlike,
        required=True,
        help="Path to the ATEPP corpus directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Pathlike,
        default=None,
        help="Path to the output directory for manifests.",
    )
    
    args = parser.parse_args()
    
    prepare_atepp(
       corpus_dir=args.corpus_dir,
       output_dir=args.output_dir
    )