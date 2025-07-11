import librosa
import pretty_midi
import scipy.io.wavfile
import numpy as np
import os
import argparse
import shutil
import glob
from tqdm import tqdm
import logging
import multiprocessing
SAMPLE_RATE = 32000

logging.basicConfig(filename='data_prepare.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
logger = logging.getLogger('data_prepare_logger')

def seg_midi(midi, start, end):
    # Create a new MIDI object for the segment
    segment_midi = pretty_midi.PrettyMIDI()
    # Set up instruments by copying from the original MIDI
    for instrument in midi.instruments:
        # Create a new instrument instance for the segment
        new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        notes = instrument.notes
        notes.sort(key=lambda x: x.start)
        # Handle notes
        for note in notes:
            if note.start < start and note.end > start:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=0,
                    end=note.end - start
                )
                new_instrument.notes.append(new_note)
            
            
            if start <= note.start and note.end < end:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - start,
                    end=note.end-start
                )
                
                new_instrument.notes.append(new_note)
            
            if note.start < end and note.end > end:
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start - start,
                    end=end - start
                )
                
            if note.start > end:
                break
            
        # Handle control changes
        control_changes = instrument.control_changes
        control_changes.sort(key=lambda x: x.time)
        for control_change in control_changes:
            if start <= control_change.time < end:
                new_control_change = pretty_midi.ControlChange(
                    number=control_change.number,
                    value=control_change.value,
                    time=control_change.time - start
                )
                new_instrument.control_changes.append(new_control_change)
            
            if control_change.time > end:
                break
        
        # Add the instrument to the segment MIDI object if it contains any notes or control changes
        if new_instrument.notes or new_instrument.control_changes:
            segment_midi.instruments.append(new_instrument)
            
    # Combine all instruments into a single instrument
    if len(segment_midi.instruments) > 1:
        combined_instrument = pretty_midi.Instrument(program=segment_midi.instruments[0].program, name="Combined Instrument")
        for inst in segment_midi.instruments:
            combined_instrument.notes.extend(inst.notes)
            combined_instrument.control_changes.extend(inst.control_changes)
            combined_instrument.pitch_bends.extend(inst.pitch_bends)
        segment_midi.instruments = [combined_instrument]
        
    # Find the latest ending time among all notes and control changes
    latest_end = 0
    for instrument in segment_midi.instruments:
        for note in instrument.notes:
            latest_end = max(latest_end, note.end)
        for control in instrument.control_changes:
            latest_end = max(latest_end, control.time)

    return segment_midi, latest_end

def seg_one(midi_wav_args_list):
    duration = midi_wav_args_list[0]
    wav_path = midi_wav_args_list[1]
    midi_path = midi_wav_args_list[2]
    perf_id = midi_wav_args_list[3]
    artist_id = midi_wav_args_list[4]
    data_path = midi_wav_args_list[5]
    output_path = midi_wav_args_list[6]
    split = midi_wav_args_list[7]
    
    wav_dir = os.path.join(data_path, wav_path)
    midi_dir = os.path.join(data_path, midi_path)
    
    midi = pretty_midi.PrettyMIDI(midi_dir)
    
    midi_segment_dir = os.path.join(output_path, f"midi_seg/{split}", str(perf_id).zfill(5) + f"_{artist_id}_0.midi")
    wav_segment_dir = os.path.join(output_path, f"wav_seg/{split}", str(perf_id).zfill(5) + f"_{artist_id}_0.wav")
    if (os.path.isfile(midi_segment_dir)) and (os.path.isfile(wav_segment_dir)):
        logger.info(f"Skipping {str(perf_id).zfill(5)}")
        return
    
    np.random.seed(42)
    min_duration = 15
    max_duration = 20

    # Generate segment points
    seg_points = [0]
    while seg_points[-1] < duration:
        next_point = seg_points[-1] + np.random.uniform(min_duration, max_duration)
        if next_point < duration:
            seg_points.append(next_point)
        else:
            break

    # Calculate segment lengths and counts
    seg_lens = [seg_points[i+1] - seg_points[i] for i in range(len(seg_points)-1)]
    seg_lens.append(duration - seg_points[-1])
    counts = [0] * len(seg_points)  # Placeholder counts since we're not using note counts
    
    
    os.makedirs(os.path.join(output_path, f"seg_info/{split}"), exist_ok=True)
    np.savez(os.path.join(output_path, f"seg_info/{split}", str(perf_id).zfill(5) + f".npz"), seg_points=seg_points, counts=counts, seg_lens=seg_lens)
    
    
    wav = librosa.core.load(wav_dir, sr=SAMPLE_RATE)[0]
    wav_seg_points = [int(k * SAMPLE_RATE) for k in seg_points]
    

    for i in range(len(seg_points) - 1):
        try:
            assert len(seg_points) == len(seg_lens)
            assert len(seg_points) == len(counts) 
        
        except AssertionError as error:
            logging.error(f"Assertion error: {error}")
            logging.info(f"seg_points: {len(seg_points)}\nseg_lens: {len(seg_lens)}\ncounts: {len(seg_points)}")
            logging.info(f"seg_points: {seg_points}")
            logging.info(f"seg_lens: {seg_lens}")
            logging.info(f"counts: {counts}")
            raise
        
        if seg_lens[i] < 3:
            if i != len(seg_points) - 1:
                logger.warning(f"Seg {i} from {seg_points[i]}s to {seg_points[i+1]}s is only {seg_lens[i]} seconds, which is too short, so skip for performance {perf_id}")
            continue
        
        midi_segment_dir = os.path.join(output_path, f"midi_seg/{split}", str(perf_id).zfill(5) + f"_{artist_id}_{i}.midi")
        wav_segment_dir = os.path.join(output_path, f"wav_seg/{split}", str(perf_id).zfill(5) + f"_{artist_id}_{i}.wav")
        
        
        wav_segment = wav[wav_seg_points[i]:wav_seg_points[i+1]]
        start = seg_points[i]
        end = seg_points[i + 1]
        
        segment_midi, _ = seg_midi(midi, start, end)
        
        # Save the segment MIDI file
        os.makedirs(os.path.dirname(midi_segment_dir), exist_ok=True)
        segment_midi.write(midi_segment_dir)
        
        os.makedirs(os.path.dirname(wav_segment_dir),  exist_ok=True)
        scipy.io.wavfile.write(wav_segment_dir, SAMPLE_RATE, wav_segment)
        
    logger.info('split uttid: {}'.format(perf_id))
    
def seg_one_midi_only(midi_wav_args_list):
    duration = midi_wav_args_list[0]
    midi_path = midi_wav_args_list[2]
    basename = midi_wav_args_list[3]
    data_path = midi_wav_args_list[5]
    output_path = midi_wav_args_list[6]
    split = midi_wav_args_list[7]
    
    midi_dir = os.path.join(data_path, midi_path)
    midi = pretty_midi.PrettyMIDI(midi_dir)
    
    midi_segment_dir = os.path.join(output_path, f"midi_seg/{split}", basename + f"_0.midi")

    if (os.path.isfile(midi_segment_dir)):
        logger.info(f"Skipping {basename}")
        return
    
    np.random.seed(42)
    min_duration = 15
    max_duration = 20

    # Generate segment points
    seg_points = [0]
    while seg_points[-1] < duration:
        next_point = seg_points[-1] + np.random.uniform(min_duration, max_duration)
        if next_point < duration:
            seg_points.append(next_point)
        else:
            break

    # Calculate segment lengths and counts
    seg_lens = [seg_points[i+1] - seg_points[i] for i in range(len(seg_points)-1)]
    seg_lens.append(duration - seg_points[-1])
    counts = [0] * len(seg_points)  # Placeholder counts since we're not using note counts
    
    # os.makedirs(os.path.join(output_path, f"seg_info/{split}"), exist_ok=True)
    # np.savez(os.path.join(output_path, f"seg_info/{split}", basename + f".npz"), seg_points=seg_points, counts=counts, seg_lens=seg_lens)
    

    for i in range(len(seg_points) - 1):

        if seg_lens[i] < 3:
            if i != len(seg_points) - 1:
                logger.warning(f"Seg {i} from {seg_points[i]}s to {seg_points[i+1]}s is only {seg_lens[i]} seconds, which is too short, so skip for performance {perf_id}")
            continue
        
        midi_segment_dir = os.path.join(output_path, basename + f"_{i}.midi")        
        start = seg_points[i]
        end = seg_points[i + 1]
        
        segment_midi, _ = seg_midi(midi, start, end)
        
        # Save the segment MIDI file
        os.makedirs(os.path.dirname(midi_segment_dir), exist_ok=True)
        segment_midi.write(midi_segment_dir)
    logger.info('split uttid: {}'.format(basename))
   
def seg_one_score(midi_wav_args_list):
    duration = midi_wav_args_list[0]
    wav_path = midi_wav_args_list[1]
    midi_path = midi_wav_args_list[2]
    perf_id = midi_wav_args_list[3]
    artist_id = midi_wav_args_list[4]
    data_path = midi_wav_args_list[5]
    output_path = midi_wav_args_list[6]
    split = midi_wav_args_list[7]
    midi_seg_points = midi_wav_args_list[8]
    wav_seg_points = midi_wav_args_list[9]

    # wav_path = row['midi_path'].replace(".mid", ".mp3")
    wav_dir = os.path.join(data_path, wav_path)
    wav_seg_points = [int(k * SAMPLE_RATE) for k in wav_seg_points]
    
    midi_dir = os.path.join(data_path, midi_path)
    midi = pretty_midi.PrettyMIDI(midi_dir)
    
    midi_segment_dir = os.path.join(output_path, f"midi_seg/{split}", str(perf_id).zfill(5) + f"_{artist_id}_0.midi")
    wav_segment_dir = os.path.join(output_path, f"wav_seg/{split}", str(perf_id).zfill(5) + f"_{artist_id}_0.wav")
    if (os.path.isfile(midi_segment_dir)) and (os.path.isfile(wav_segment_dir)):
        logger.info(f"Skipping {str(perf_id).zfill(5)}")
        return

    # seg_points, counts, seg_lens = get_seg_points(duration, midi)
    
    wav = librosa.core.load(wav_dir, sr=SAMPLE_RATE)[0]
    
    # logger.info(f"Segment points were found. The minimum segment duration: {min(seg_lens[:-1])}, the maxmum notes number: {max(counts)}")
    logger.info('start split uttid: {}'.format(perf_id))
    
    for i in range(len(wav_seg_points) - 1):
        if (wav_seg_points[i] != int(-1 * SAMPLE_RATE)) & (wav_seg_points[i+1] != int(SAMPLE_RATE * -1)):
            # Remove the segments longer than 20 seconds
            wav_segment = wav[wav_seg_points[i]:wav_seg_points[i + 1]]
        
            midi_segment_dir = os.path.join(output_path, f"midi_seg/{split}", str(perf_id).zfill(5) + f"_{artist_id}_{i}.midi")
            wav_segment_dir = os.path.join(output_path, f"wav_seg/{split}", str(perf_id).zfill(5) + f"_{artist_id}_{i}.wav")
            
            # if i == len(wav_seg_points) - 2:
            #     wav_segment = wav[wav_seg_points[i]:]
            #     start = midi_seg_points[i]
            #     end = midi.get_end_time()
            # else:
            start = midi_seg_points[i]
            end = midi_seg_points[i + 1]
        
            segment_midi = seg_midi(midi, start, end)
            
            if segment_midi.instruments == [] or len(segment_midi.instruments[0].notes) > 254:
                continue
            
             # Save the segment MIDI file
            os.makedirs(os.path.dirname(midi_segment_dir), exist_ok=True)
            segment_midi.write(midi_segment_dir)
            
            os.makedirs(os.path.dirname(wav_segment_dir), exist_ok=True)
            scipy.io.wavfile.write(wav_segment_dir, SAMPLE_RATE, wav_segment)
        
    logger.info('split uttid: {}'.format(perf_id))
      
def prepare_infer_prompts(midi_dir, wav_dir, start, end):
    wav = librosa.core.load(wav_dir, sr=SAMPLE_RATE)[0]
    midi = pretty_midi.PrettyMIDI(midi_dir)
    midi_prompt, _ = seg_midi(midi, start, end)
    audio_prompt = wav[int(start * SAMPLE_RATE): int(end*SAMPLE_RATE)]  
    return midi_prompt, audio_prompt

def concat_midi_files(midi_file1, midi_file2, output_file):
    # Load the two MIDI files
    midi1 = pretty_midi.PrettyMIDI(midi_file1)
    midi2 = pretty_midi.PrettyMIDI(midi_file2)
    
    # Find the total length of the first MIDI file
    end_time_midi1 = midi1.get_end_time()
    
    # Shift all events of the second MIDI file by the length of the first MIDI file
    for instrument in midi2.instruments:
        for note in instrument.notes:
            note.start += end_time_midi1
            note.end += end_time_midi1
        midi1.instruments[0].notes.extend(instrument.notes)
        for control_change in instrument.control_changes:
            control_change.time += end_time_midi1
        midi1.instruments[0].control_changes.extend(instrument.control_changes)
        for pitch_bend in instrument.pitch_bends:
            pitch_bend.time += end_time_midi1
        midi1.instruments[0].pitch_bends.extend(instrument.pitch_bends)
    
    # Merge instruments of midi2 into midi1
    
    # Write the concatenated MIDI file to the output path
    midi1.write(output_file)
    del midi1
    del midi2

def get_midi_audio_pairs(data_path, out_path, audio=True):
    midi_files = glob.glob(f"{data_path}/**/*.mid", recursive=True)
    midi_files += glob.glob(f"{data_path}/**/*.midi", recursive=True)

    midi_wav_args_list = []
    for midi_file in tqdm(midi_files):
        dataset = midi_file.split("/")[-3] 
        basename = os.path.basename(midi_file).split(".")[0]
        duration = pretty_midi.PrettyMIDI(midi_file).get_end_time()

        if audio:
            if dataset == "ATEPP":
                audio_file = midi_file.replace(".mid", ".mp3").replace("midi", "audio")
            elif dataset == "pijama":
                audio_file = midi_file.replace(".midi", ".mp3").replace("midi", "audio")
            elif dataset == "maestro":
                audio_file = midi_file.replace(".mid", ".mp3").replace("midi", "audio")
            else:
                audio_file = midi_file.replace(".mid", ".mp3").replace("midi", "audio")
                dataset = "test"

            if not os.path.exists(audio_file):
                logger.warning(f"Audio file {audio_file} does not exist for {basename}. Skipping.")
                continue
            
        else:
            audio_file = None
            dataset = "test"
       
        midi_wav_args_list.append([
            duration, audio_file, midi_file, basename, 0,
            data_path, os.path.join(out_path, dataset), "test"
        ])
        
    return midi_wav_args_list

def concat_prompts_with_original(data_path: str, dataset: str, prompt_dir: str):
    """
    Concatenates prompt MIDI files with their corresponding original MIDI files.

    Args:
        data_path (str): Root directory containing dataset folders.
        dataset (str): Name of the dataset (e.g., "ATEPP").
        prompt_dir (str): Directory containing prompt MIDI files.
    """
    logger.info(f"Concatenating prompt MIDIs with original for dataset: {dataset}")

    prompt_files = glob.glob(f"{prompt_dir}/*s.midi")
    target_dir = os.path.join(data_path, dataset, "concat_midi")
    org_midi_dir = os.path.join(data_path, dataset, "midi_seg", "test")
    os.makedirs(target_dir, exist_ok=True)

    for prompt_file in tqdm(prompt_files):
        basename = os.path.basename(prompt_file).split(".")[0]
        audio_file = prompt_file.replace(".midi", ".wav")
        midi_file = os.path.join(org_midi_dir, f"{basename}.midi")  # Adjust if needed
        output_file = os.path.join(target_dir, f"{basename}_cat.midi")

        if os.path.exists(output_file):
            continue

        try:
            shutil.copy(audio_file, output_file.replace(".midi", ".wav"))
            concat_midi_files(prompt_file, midi_file, output_file)
        except Exception as e:
            logger.warning(f"Failed to concat {basename}: {e}")

def prepare_segments(data_path, out_path, audio=True):
    logger.info("Stage 0: Generating MIDI-Audio Pairs")

    args_list = get_midi_audio_pairs(data_path, out_path, audio=audio)
    print(args_list)
    with multiprocessing.Pool(processes=8) as pool:
        if audio:
            for _ in tqdm(pool.imap(seg_one, args_list), total=len(args_list)):
                pass
        else:
            for _ in tqdm(pool.imap(seg_one_midi_only, args_list), total=len(args_list)):
                pass
            
def process_for_batch_inference(data_path, datasets=["ATEPP", "maestro", "pijama"]):
    logger.info("Stage 1: Preparing prompt and concatenated MIDI")
    
    for dataset in datasets:
        midi_paths = glob.glob(f"{data_path}/{dataset}/midi_seg/test/*.midi")
        prompt_dir = f"{data_path}/{dataset}/prompt_gt"
        os.makedirs(prompt_dir, exist_ok=True)

        for midi_path in tqdm(midi_paths):
            wav_path = midi_path.replace("midi", "wav")
            basename = os.path.basename(midi_path).split(".")[0]
            midi_prompt_path = os.path.join(prompt_dir, f"{basename}_prompt_0s.midi")
            audio_prompt_path = os.path.join(prompt_dir, f"{basename}_prompt_0s.wav")

            if os.path.exists(midi_prompt_path) and os.path.exists(audio_prompt_path):
                continue

            midi_prompt, audio_prompt = prepare_infer_prompts(midi_path, wav_path, 0, 3)
            midi_prompt.write(midi_prompt_path)
            scipy.io.wavfile.write(audio_prompt_path, SAMPLE_RATE, audio_prompt)

            shutil.copy(wav_path, audio_prompt_path.replace("_prompt_0s", ""))
            shutil.copy(midi_path, midi_prompt_path.replace("_prompt_0s", ""))

    logger.info("Stage 2: Concatenating prompts with original MIDI files")
    for dataset in datasets:
        prompt_dir = f"{data_path}/{dataset}/prompt_gt"
        concat_prompts_with_original(data_path, dataset, prompt_dir)

def process_for_performance_inference_colab(data_path, out_path, prompt_midi_dir, prompt_wav_dir):
    logger.info("Preparing prompts and concat for individual performance")
    midi_paths = sorted(
        glob.glob(f"{data_path}/**/*.midi", recursive=True),
        key=lambda x: int(x.split("/")[-1].split("_")[-1].split(".")[0])
    )

    for midi_path in tqdm(midi_paths):
        perf_id = os.path.basename(midi_path).split(".")[0].split("_")[0]
        # indiv_out = os.path.join(out_path, perf_id + "_prompt")
        cat_dir = os.path.join(out_path, f"cat_{perf_id}")
        # os.makedirs(indiv_out, exist_ok=True)
        os.makedirs(cat_dir, exist_ok=True)

        basename = os.path.basename(midi_path).split(".")[0]

        # midi_prompt_path = os.path.join(indiv_out, f"prompt.midi")
        # audio_prompt_path = os.path.join(indiv_out, f"prompt.wav")
        cat_path = os.path.join(cat_dir, f"{basename}_cat.midi")

        # midi_prompt, audio_prompt = prepare_infer_prompts(prompt_midi_dir, prompt_wav_dir, 0, 3)
        # midi_prompt.write(midi_prompt_path)
        # scipy.io.wavfile.write(audio_prompt_path, SAMPLE_RATE, audio_prompt)

        try:
            concat_midi_files(prompt_midi_dir, midi_path, cat_path)
        except Exception as e:
            logger.warning(f"Concat failed for {basename}: {e}")
            continue
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI-Audio Data Preprocessing")
    subparser = parser.add_subparsers(dest="mode", required=True) 
    
    seg_parser = subparser.add_parser("prepare_segments", help="Prepare segments from MIDI and audio files")
    seg_parser.add_argument("--data_path", type=str, required=True, help="Input data directory")
    seg_parser.add_argument("--out_path", type=str, required=True, help="Output directory")
    seg_parser.add_argument("--audio", action="store_true", help="Whether to process audio files")
    
    batch_parser = subparser.add_parser("process_for_batch_inference", help="Process for batch inference")
    batch_parser.add_argument("--data_path", type=str, required=True, help="Input data directory")
    batch_parser.add_argument("--datasets", type=list, required=True, help="Output directory")
    
    performance_parser = subparser.add_parser("process_for_performance_inference", help="Process for individual performance inference")
    performance_parser.add_argument("--data_path", type=str, required=True, help="Input data directory")
    performance_parser.add_argument("--out_path", type=str, required=True, help="Output directory")
    performance_parser.add_argument("--prompt_midi_dir", type=str, required=True, help="Prompt MIDI directory")
    performance_parser.add_argument("--prompt_wav_dir", type=str, required=True, help="Prompt WAV directory")
    
    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Data Path: {args.data_path}")

    if args.mode == 'prepare_segments':
        prepare_segments(args.data_path, args.out_path, args.audio)
    elif args.mode == 'process_for_batch_inference':
        process_for_batch_inference(args.data_path, args.datasets)
    else:
        process_for_performance_inference_colab(args.data_path, args.out_path, args.prompt_midi_dir, args.prompt_wav_dir)
