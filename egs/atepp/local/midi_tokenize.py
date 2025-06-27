import os
from expression_tokenizer import ExpressionTok
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse

BOS_ID = 1
EOS_ID = 2

TICKS_PER_BEAT = 96

TOKENIZER_PARAMS = {
        "pitch_range": (21, 109),
        "beat_res": {(0, 12):TICKS_PER_BEAT},
        "num_velocities": 64,
        "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
        "use_chords": False,
        "use_rests": False,
        "use_tempos": False,
        "use_time_signatures": False,
        "use_programs": False,
        "tempo_range": (40, 250),  # (min, max)
    }

def process_with_expression_tokenizer(data_folder, target_folder):
    tokenizer = ExpressionTok(params=Path("midi-valle-tokenizer.json"))
    corpus_dir = Path(data_folder)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    transcript_dir = corpus_dir / target_folder
    transcript_files = transcript_dir.rglob("*.midi")
    slens = {}
    
    for file in tqdm(transcript_files):
        if os.path.isfile(file.parent / (str(file.stem) + ".npy")):
            continue
        try:
            sequence = tokenizer(file)[0].ids
        except:
            os.remove(file)
            # os.remove(str(file).replace("midi_seg", "wav_seg").replace("midi", "wav"))
            print(f"Cannot Open: {file.stem}")
            # print(file)
            continue 
        
        # If the sequence is empty, remove the file
        if len(sequence) == 0:
            os.remove(file)
            os.remove(str(file).replace("midi_seg", "wav_seg").replace("midi", "wav"))
            print(f"File Removed: {file.stem}")
            continue
        
        category_length = len(sequence[0])
        
        # Add BOS and EOS tokens    
        bos_row = [[BOS_ID] * category_length]
        eos_row = [[EOS_ID] * category_length]
        sequence = bos_row + sequence + eos_row
        slens[str(file.stem)] = len(sequence)
        
        np.save(file.parent / (str(file.stem) + ".npy"), sequence)
        
    # np.savez("valle-stats.npz", slens=slens)
    slens_values = [v for v in slens.values()]
    print(f"The minimum sequenc lens: {min(slens_values)}")
    print(f"The maximum sequenc lens: {max(slens_values)}")
    print(f"average lens: {sum(slens_values)/len(slens_values)}")
    print((np.asarray(slens_values)< 100).sum())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIDI files with ExpressionTok")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder containing MIDI files")
    parser.add_argument("--target_folder", type=str, default="midi_seg", help="Target folder to save processed files")
    
    args = parser.parse_args()
    
    process_with_expression_tokenizer(args.data_folder, args.target_folder)
    print("Processing completed.")