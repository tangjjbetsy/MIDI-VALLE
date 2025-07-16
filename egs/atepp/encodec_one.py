from audiocraft.models import CompressionModel
from audiocraft.data.audio_utils import convert_audio
import torchaudio
import argparse
import torch
import os

def reconstruct(audio_file, output_file, model_path=None):
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    
    if model_path:
        model = CompressionModel.get_pretrained(model_path)
    else:
        model = CompressionModel.get_pretrained('compression_32khz_new.bin')

    wav, sr = torchaudio.load(audio_file)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    
    model.eval()
    
    with torch.no_grad():
        codes, scale = model.encode(wav)
        frames = model.decode(codes, scale)

    torchaudio.save(output_file, frames[0].detach(), 32000)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file path",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output audio file path",
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the pre-trained model file. If not provided, uses the default model.",
    )
    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    reconstruct(args.input, args.output, args.model_path)
