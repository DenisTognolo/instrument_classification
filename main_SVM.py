# Example of Support Vector Machine

from instrument_classificator import *

def main(folder, dur, rateo, feature_type, frame_length, hop_length, n_mels, order, max_iter):

  exp = "SVM - " + feature_type + " - fl: " + str(frame_length) + " - hl: " + str(hop_length) + " - nm: " + str(n_mels) + " - ord: " + str(order) + " - mi: " +  str(max_iter)
  print("STARTING EXPERIMENT: " + exp + " ...")

  train_x, train_y, test_x, test_y, labels = import_dataset(folder, dur, rateo)

  print("DATASET LOADED!")

  features = extract_feature(feature_type, train_x, frame_length, hop_length, n_mels) 
  test_features = extract_feature(feature_type, test_x, frame_length, hop_length, n_mels) 

  print("FEATURE EXTRACTED!")
  # print(features.shape)
  
  models = train_SVM(features, train_y, order, max_iter)

  print("MODEL TRAINED!")
  
  conf_mat = eval_SVM_models(models, test_features, test_y)
  accuracy, precision, recall, fscore = get_metrics(conf_mat)

  print("EVALUATION DONE!")

  save_metrics(exp, accuracy, precision, recall, fscore, labels)
  save_conf_mat(exp, conf_mat, labels)

  print("RESULTS SAVED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Support Vector Machine + Mel-Frequency Cepstral Coefficients')
    parser.add_argument("--folder", type=str, default='dataset', help="Folder of the dataset. Must contain one subfolder for each class")
    parser.add_argument("--dur", type=float, default=1, help="Duration of the sample in second")
    parser.add_argument("--rateo", type=float, default=0.75, help="Rateo [0 1] between training and test set")
    parser.add_argument("--feature_type", type=str, default="signal", help="Type of feature to consider: [signal, STFT, SC, MSTFT, MFCC]")
    parser.add_argument("--frame_length", type=int, default=1024, help="Size of the FFT, which will also be used as the window length")
    parser.add_argument("--hop_length", type=int, default=512, help="Step or stride between windows. If the step is smaller than the window length, the windows will overlap")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of Mel bands to generate")
    parser.add_argument("--order", type=int, default=3, help="Order of the polynamial function to consider as kernel (linear if order=1)")
    parser.add_argument("--max_iter", type=int, default='10000', help="Hard limit on iterations within solver")

    args = parser.parse_args()

    folder = args.folder
    dur = args.dur
    rateo = args.rateo
    feature_type = args.feature_type
    frame_length = args.frame_length
    hop_length = args.hop_length
    n_mels = args.n_mels
    order = args.order
    max_iter = args.max_iter
    
    main(folder, dur, rateo, feature_type, frame_length, hop_length, n_mels, order, max_iter)