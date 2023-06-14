# Example of K-Nearest Neighbors

from instrument_classificator import *

def main(folder, dur, rateo, feature_type, frame_length, hop_length, n_mels, n_neigh):

  exp = "KNN - " + feature_type + " - fl: " + str(frame_length) + " - hl: " + str(hop_length) + " - nm: " + str(n_mels) + " - nn: " + str(n_neigh)
  print("STARTING EXPERIMENT: " + exp + " ...")

  train_x, train_y, test_x, test_y, labels = import_dataset(folder, dur, rateo)

  print("DATASET LOADED!")
  
  features = extract_feature(feature_type, train_x, frame_length, hop_length, n_mels) 
  test_features = extract_feature(feature_type, test_x, frame_length, hop_length, n_mels) 

  print("FEATURE EXTRACTED!")
  # print(features.shape)

  model = train_KNN(features, train_y, n_neigh)

  print("MODEL TRAINED!")

  conf_mat = eval_KNN_model(model, test_features, test_y)
  accuracy, precision, recall, fscore = get_metrics(conf_mat)

  print("EVALUATION DONE!")

  save_metrics(exp, accuracy, precision, recall, fscore, labels)
  save_conf_mat(exp, conf_mat, labels)

  print("RESULTS SAVED!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='K-Nearest Neighbors + Mel-Frequency Cepstral Coefficients')
    parser.add_argument("--folder", type=str, default='dataset', help="Folder of the dataset. Must contain one subfolder for each class")
    parser.add_argument("--dur", type=float, default=1, help="Duration of the sample in second")
    parser.add_argument("--rateo", type=float, default=0.75, help="Rateo [0 1] between training and test set")
    parser.add_argument("--feature_type", type=str, default="signal", help="Type of feature to consider: [signal, STFT, SC, MSTFT, MFCC]")
    parser.add_argument("--frame_length", type=int, default=1024, help="Size of the FFT, which will also be used as the window length")
    parser.add_argument("--hop_length", type=int, default=512, help="Step or stride between windows. If the step is smaller than the window length, the windows will overlap")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of Mel bands to generate")
    parser.add_argument("--n_neigh", type=int, default=5, help="Number of neighbors to consider during KNN")

    args = parser.parse_args()

    folder = args.folder
    dur = args.dur
    rateo = args.rateo
    feature_type = args.feature_type
    frame_length = args.frame_length
    hop_length = args.hop_length
    n_mels = args.n_mels
    n_neigh = args.n_neigh
    
    main(folder, dur, rateo, feature_type, frame_length, hop_length, n_mels, n_neigh)