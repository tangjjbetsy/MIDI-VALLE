import glob
import os
import shutil

segments = glob.glob("/data/scratch/acw555/ATEPP-valle-large/midi_seg/test/*.midi", recursive=True)

segments = segments[0:2000:100]

for i in segments:
    basename = os.path.basename(i)
    shutil.copy(i, f"/data/scratch/acw555/ATEPP-valle-large/download/{basename}")
    shutil.copy(i.replace("midi", "wav"), f"/data/scratch/acw555/ATEPP-valle-large/download/{basename.replace('.midi', '.wav')}")