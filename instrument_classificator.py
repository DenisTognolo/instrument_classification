import argparse, os
import numpy as np

# audio lib
import librosa

# visualization
import matplotlib.pyplot as plt

# machine learning
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# DATA PROCESSING

def import_sample(sample_file, dur):

  sample, sr = librosa.load(sample_file)
  sample, index = librosa.effects.trim(sample, top_db=20)   #Remove silence
  sample = sample[0:int(dur*sr)]
  sample/=np.max(np.abs(sample))

  label = sample_file.split("/")[1]

  # List to np.array (+ remove not finite values and scale)
  sample = np.nan_to_num(np.array(sample))
  return sample, label

def import_dataset(folder, dur, rateo):
  directory_files = os.listdir(folder)
  labels = [f.name for f in os.scandir(folder)]

  x = []
  y = []
  i = 0

  for lab in labels:
    directory_files = os.listdir(folder + "/" + lab)

    for file in directory_files:
        sample, sr = librosa.load(folder + "/" + lab + "/" + file)
        sample, index = librosa.effects.trim(sample, top_db=20)   #Remove initial silence
        sample = sample[0:int(dur*sr)]                            #Set desired duration
        sample/=np.max(np.abs(sample))                            #Normalize

        x.append(sample)                                          #Sample
        y.append(i)                                               #Class GT
    i+=1

  n_x = len(x)
  n_y = len(set(y))
  print(str(n_x) + " files imported, for a total of " + str(n_y) + " classes:") 
  print(labels)

  # Pad to have same duration sample
  dur=0
  for xi in x:
    if len(xi)>dur:
      dur=len(xi)

  X = []
  for xi in x:
    X.append(librosa.util.fix_length(xi, size=dur))
  x=X

  # Create Train and Test Sets
  n = len(x)
  tr_n = n*rateo

  test_x = []
  test_y = []

  while n > tr_n:
    i = np.random.randint(n)
    test_x.append(x.pop(i))
    test_y.append(y.pop(i))
    n-=1

  train_x = x
  train_y = y

  # List to np.array (+ remove not finite values and scale)
  
  train_x = np.nan_to_num(np.array(x))
  train_y = np.nan_to_num(np.array(y))

  test_x = np.nan_to_num(np.array(test_x))
  test_y = np.nan_to_num(np.array(test_y))

  # (nobj, nfeat) = train_x.shape
  # print(str(nobj) + " samples in the training set, with " + str(nfeat) + " frames.")

  return train_x, train_y, test_x, test_y, labels

# FEATURE EXTRACTION

def extract_feature(type, datas, frame_length, hop_length, n_mels):
  if type == 'signal':
    return datas
  elif type == 'STFT':
    return extract_STFT(datas, frame_length,  hop_length)
  elif type == 'SC':
    return extract_SC(datas, frame_length, hop_length)
  elif type == 'MSTFT':
    return extract_MSTFT(datas, frame_length, hop_length, n_mels)
  elif type == 'MFCC':
    return extract_MFCC(datas, frame_length, hop_length, n_mels)
  else:
    raise("Please enter a valid feature extractor: [signal, SC, STFT, MSTFT, MFCC]")

def extract_STFT(datas, frame_length, hop_length):
  features = librosa.stft(y=datas, n_fft=frame_length, hop_length=hop_length, win_length=frame_length)
  features = np.abs(features)
  features = librosa.amplitude_to_db(features, ref=np.max)
  
  return features

def extract_SC(datas, frame_length, hop_length):
  
  features = librosa.feature.spectral_centroid(y=datas, n_fft=frame_length, hop_length=hop_length, win_length=frame_length)
  features = np.asarray(features)  

  if len(features.shape) > 2:
    features = features.reshape(features.shape[0], -1)
  else:
    features = features.reshape(-1)

  return features

def extract_MSTFT(datas, frame_length, hop_length, n_mels):
  features = librosa.feature.melspectrogram(y=datas, center=False, n_fft=frame_length, hop_length=hop_length, win_length=frame_length, n_mels=n_mels, fmax=8000) 
  features = librosa.amplitude_to_db(features, ref=np.max)
  
  return features

def extract_MFCC(datas, frame_length, hop_length, n_mels):
  features = librosa.feature.mfcc(y=datas, n_fft=frame_length, hop_length=hop_length, win_length=frame_length, n_mfcc=n_mels)
  features = librosa.amplitude_to_db(features, ref=np.max)

  return features

# MODELING

def train_SVM(x, y, order, max_iter):

  if len(x.shape) > 2:
    x = x.reshape(x.shape[0], -1)

  # x = minmax_scale(x, axis=0)

  if order == 1:
    kernel = 'linear'
  else:
    kernel = 'poly'

  n = len(set(y))
  models = [SVC(kernel=kernel, degree=order, max_iter=max_iter, probability=True) for i in range(n)]

  for i in range(n):
    models[i].fit(x, y==i)

  return models 

def train_KNN(x, y, n_neigh):

  if len(x.shape) > 2:
    x = x.reshape(x.shape[0], -1)

  model = KNeighborsClassifier(n_neighbors=n_neigh)
  model.fit(x, y)
  return model

# TESTING

def single_predict_SVM(models, x):

  if len(x.shape) > 2:
    x = x.reshape(x.shape[0], -1)

  # x = minmax_scale(x, axis=0)

  n = len(models)
  predicted_scores = []
  for i in range(n):
    predicted_scores.append(models[i].predict_proba(x))
  
  predicted_scores = np.asarray(predicted_scores)
  y_pred = np.argmax(predicted_scores[:,:,1] , axis=0)[0]                #take the maximum probabilities to be in

  return y_pred

# EVALUATION

def eval_SVM_models(models, x, y):

  if len(x.shape) > 2:
    x = x.reshape(x.shape[0], -1)

  # x = minmax_scale(x, axis=0)

  n = len(models)
  predicted_scores = []
  for i in range(n):
    predicted_scores.append(models[i].predict_proba(x))

  predicted_scores = np.asarray(predicted_scores)
  y_pred = np.argmax(predicted_scores[:,:,1], axis=0)                  #take the maximum probabilities to be in

  conf_mat = confusion_matrix(y, y_pred)

  return conf_mat

def eval_KNN_model(model, x, y):

  if len(x.shape) > 2:
    x = x.reshape(x.shape[0], -1)

  y_pred = model.predict(x)  
  conf_mat = confusion_matrix(y, y_pred)
  return conf_mat

def get_metrics(conf_mat):
  
  n, m = conf_mat.shape

  accuracy = np.zeros(n)
  precision = np.zeros(n)
  recall = np.zeros(n)
  fscore = np.zeros(n)

  for i in range(n):

      TP = conf_mat[i,i]
      FP = sum(conf_mat[:,i]) - TP
      FN = sum(conf_mat[i,:]) - TP
      TN = sum(sum(conf_mat)) - TP -  FP - FN

      accuracy[i] = (TN+TP) / (TN+FP+FN+TP)
      precision[i]  = TP / (TP+FP)
      recall[i] = TP / (TP+FN)
      fscore[i] = 2*(precision[i]*recall[i])/(precision[i]+recall[i])

  return accuracy, precision, recall, fscore

# VISUALIZATION

def show_feature(type, signal, label):
  if type == 'signal':
    show_signal(signal, label)
  elif type == 'STFT':
    show_STFT(signal, label)
  elif type == 'SC':
    show_SC(signal, label)
  elif type == 'MSTFT':
    show_MSTFT(signal, label)  
  elif type == 'MFCC':
    show_MFCC(signal, label)
  else:
    raise("Please enter a valid feature extractor: [signal, SC, STFT, MSTFT, MFCC]")

def show_signal(signal, label):
  fig, ax = plt.subplots()
  librosa.display.waveshow(signal, alpha=0.8)
  ax.set_title("Loudness signal for a " + label + "...")
  ax.set_xlabel("sec")
  ax.set_ylabel("dB")
  plt.show()

def show_STFT(signal, label):
  fig, ax = plt.subplots()
  img = librosa.display.specshow(signal, y_axis='linear', x_axis='time', ax=ax, fmax=8000)
  ax.set_title("Power spectrogram for a " + label + "...")
  fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.show()

def show_SC(signal, label):
  t = librosa.frames_to_time(range(len(signal)))
  fig, ax = plt.subplots()
  plt.plot(t, signal, color='r')
  ax.set_title("Spectral centroid for a " + label + "...")
  ax.set_xlabel("sec")
  ax.set_ylabel("hz")
  plt.show()

def show_MSTFT(signal, label):
  fig, ax = plt.subplots()
  img = librosa.display.specshow(signal, y_axis='linear', x_axis='time', ax=ax, fmax=8000)
  ax.set_title("Mel-frequency spectrogram for a " + label + "...")
  fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.show()

def show_MFCC(signal, label):
  fig, ax = plt.subplots()
  img = librosa.display.specshow(signal, y_axis='linear', x_axis='time', ax=ax, fmax=8000)
  ax.set_title("Mel-frequency cepstral coefficients for a " + label + "...")
  fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.show()

def show_metrics(accuracy, precision, recall, fscore, labels): 
  for i in range(accuracy.shape[0]):
    print("Class " + labels[i] + " :: accuracy: " + str(accuracy[i].round(2)) + " | precision: " + str(precision[i].round(2)) + " | recall: " + str(recall[i].round(2)) + " | fscore: " + str(fscore[i].round(2)))
  print("\nTotal Model :: accuracy: " + str(np.mean(accuracy).round(2)) + " | precision: " + str(np.mean(precision).round(2)) + " | recall: " + str(np.mean(recall).round(2)) + " | fscore: " + str(np.mean(fscore).round(2)))

def show_conf_mat(conf_mat, labels):
  place = [i for i in range(len(labels))]

  plt.imshow(conf_mat)
  plt.colorbar()
  plt.xlabel("Predicted")
  plt.ylabel("Ground Truth")
  plt.xticks(place, labels)
  plt.yticks(place, labels)
  plt.show()

# SAVE TO FILE

def save_feature(exp, type, signal, label):
  if type == 'signal':
    save_signal(exp, signal, label)
  elif type == 'STFT':
    save_STFT(exp, signal, label)
  elif type == 'SC':
    save_SC(exp, signal, label)  
  elif type == 'MSTFT':
    save_MSTFT(exp, signal, label)  
  elif type == 'MFCC':
    save_MFCC(exp, signal, label)
  else:
    raise("Please enter a valid feature extractor: [signal, SC, STFT, MSTFT, MFCC]")

def save_signal(exp, signal, label):
  fig, ax = plt.subplots()
  librosa.display.waveshow(signal, alpha=0.8)
  ax.set_title("Loudness signal for a " + label + "...")
  ax.set_xlabel("sec")
  ax.set_ylabel("dB")
  plt.savefig("images/" + exp + ".png")

def save_STFT(exp, signal, label):
  fig, ax = plt.subplots()
  img = librosa.display.specshow(signal, y_axis='linear', x_axis='time', ax=ax, fmax=8000)
  ax.set_title("Power spectrogram for a " + label + "...\n" + exp)
  fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.savefig("images/" + exp + ".png")

def save_SC(exp, signal, label):
  t = librosa.frames_to_time(range(len(signal)))
  fig, ax = plt.subplots()
  plt.plot(t, signal, color='r')

  ax.set_title("Spectral centroid for a " + label + "...\n" + exp)
  ax.set_xlabel("sec")
  ax.set_ylabel("hz")
  plt.savefig("images/" + exp + ".png")

def save_MSTFT(exp, signal, label):
  fig, ax = plt.subplots()
  img = librosa.display.specshow(signal, y_axis='linear', x_axis='time', ax=ax, fmax=8000)
  ax.set_title("Mel-frequency spectrogram for a " + label + "...\n" + exp)
  fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.savefig("images/" + exp + ".png")

def save_MFCC(exp, signal, label):
  fig, ax = plt.subplots()
  img = librosa.display.specshow(signal, y_axis='linear', x_axis='time', ax=ax, fmax=8000)
  ax.set_title("Mel-frequency cepstral coefficients for a " + label + "...\n" + exp)
  fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.savefig("images/" + exp + ".png")

def save_metrics(exp, accuracy, precision, recall, fscore, labels): 

  f = open("results/" + exp + ".txt", "x")
  f.write(exp + ":\n\n")

  for i in range(accuracy.shape[0]):
    f.write("Class " + labels[i] + " :: accuracy: " + str(accuracy[i].round(2)) + " | precision: " + str(precision[i].round(2)) + " | recall: " + str(recall[i].round(2)) + " | fscore: " + str(fscore[i].round(2))+ "\n")
  f.write("\nTotal Model :: accuracy: " + str(np.mean(accuracy).round(2)) + " | precision: " + str(np.mean(precision).round(2)) + " | recall: " + str(np.mean(recall).round(2)) + " | fscore: " + str(np.mean(fscore).round(2)))

  f.close()

def save_conf_mat(exp, conf_mat, labels):
  place = [i for i in range(len(labels))]

  plt.imshow(conf_mat)
  plt.colorbar()
  plt.xlabel("Predicted")
  plt.ylabel("Ground Truth")
  plt.xticks(place, labels)
  plt.yticks(place, labels)
  plt.title(exp)
  plt.savefig("results/conf_mat_" + exp + ".png")
