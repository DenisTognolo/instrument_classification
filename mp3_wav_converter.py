import argparse, os
import subprocess


def main(folder_in):
  #import dataset
  directory_files = os.listdir(folder_in)
  labels = [f.name for f in os.scandir(folder_in)]

  folder_out = folder_in[:-4]
  #os.makedirs(folder_out)

  for lab in labels:
    os.makedirs(folder_out + "/" + lab)
    directory_files = os.listdir(folder_in + "/" + lab)

    for file in directory_files:
      mp3 = folder_in + "/" + lab + "/" + file
      wav = folder_out + "/" + lab + "/" + file[:-5]+".wav"
      subprocess.call(['ffmpeg', '-i', mp3, wav])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='K-Nearest Neighbors + Mel-Frequency Cepstral Coefficients')
  parser.add_argument("--folder_in", type=str, default='Philharmonia-samples_mp3', help="Folder for the dataset containing .mp3 files")

  args = parser.parse_args()

  folder_in = args.folder_in
  main(folder_in)