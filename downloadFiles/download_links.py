import subprocess

#Import data packages
import os
import sys
import glob
import pandas as pd
import sys

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


ruta_archivo_links = "Links190623.txt"
datos_link = pd.read_csv(ruta_archivo_links, sep=" ",header=None,names=["name", "link"])

print(datos_link.head())

for index, row in datos_link.iterrows():
    file_name = row['name']
    cdn_link = row['link']
    link_wget= "wget -O "+ file_name +" "+ cdn_link
    print('"'+link_wget+'"')
    runcmd('"'+link_wget+'"' , verbose = True)
