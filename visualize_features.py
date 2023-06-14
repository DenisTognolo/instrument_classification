# Feature Visulizator for a file specified sample

from instrument_classificator import *

def main(sample_file, dur, frame_length, hop_length, n_mels):

  sample, label = import_sample(sample_file, dur)

  print("SAMPLE LOADED!")

  feature_types = ["signal", "STFT", "SC", "MSTFT", "MFCC"]
  # feature_types = ["STFT", "SC"]

  for feature_type in feature_types:

    feature = extract_feature(feature_type, sample, frame_length, hop_length, n_mels) 

    print("FEATURE " + feature_type + " EXTRACTED!")

    exp = sample_file.split("/")[2] + " - " + feature_type + " - fl: " + str(frame_length) + " - hl: " + str(hop_length) + " - nm: " + str(n_mels)
    # exp = sample_file.split("/")[2] + " - " + feature_type + " - fl: " + str(frame_length) + " - hl: " + str(hop_length)
    save_feature(exp, feature_type, feature, label)
    
    print("FEATURE " + feature_type + " SAVED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Features Visulizator')
    parser.add_argument("--sample_file", type=str, default='sample', help="Location of the audio file to analyse")
    parser.add_argument("--dur", type=float, default=1, help="Duration of the sample to consider")
    parser.add_argument("--frame_length", type=int, default=1024, help="Size of the FFT, which will also be used as the window length")
    parser.add_argument("--hop_length", type=int, default=512, help="Step or stride between windows. If the step is smaller than the window length, the windows will overlap")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of Mel bands to generate")

    args = parser.parse_args()

    sample_file = args.sample_file
    dur = args.dur
    frame_length = args.frame_length
    hop_length = args.hop_length
    n_mels = args.n_mels

    main(sample_file, dur, frame_length, hop_length, n_mels)