import os
import math
import subprocess

current_directory = os.getcwd()
audio_files = [os.path.join(current_directory, file) for file in os.listdir(current_directory) if file.endswith(".wav")]

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


for file in audio_files:
     runcmd('mv '+ file +' CCv2_part_35_audios/')