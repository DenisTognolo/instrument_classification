# Music Instrument Classification with Machine Learning
This work goals to develop a classificator for music instruments using different Machine Learning techniques and compare them. 
In particular I tried four different techniques for feature extraction and two classification models, all of
them tuned properly in their parameters, in order to find the most performant solution.

Please refer to the *Report.pdf* for all the details.

## Dependencies:
```terminal
pip install numpy
pip install librosa
pip install scikit-learn
pip install matplotlib
```

## USAGE:

### MP3 to WAV converter:
*Convert all the mp3 files contained in the folder 'your_folder_mp3' (and its sub folders) into wav files and save into a new folder 'your_folder' maintaining the internal folder structure.*

```terminal
python mp3_wav_converter.py --folder_in='your_folder_mp3'
```

### Single Feature Visualizator:
*Visualize all the features (original, STFT, MSTFT, SC, MFCC) for a given audio file 'audio.wav', specifying some feature specifics such as duration, frame length, hop length, and number of mel-bands. A figure of each feature will be saved inside the images folder.*

```terminal
python visualize_features.py --sample_file='audio.wav' --dur=.. --frame_length=.. --hop_length=.. --n_mels=..
```

### Multi Feature Visualizator:
*Open the file 'run_visualization.sh' and uncomment and/or write your multiple test setup*

```terminal
. run_visualization.sh
```

### Single KNN Classifier:
*Run the training and then evaluate a model based on KNN and a certain feature to be specified. In particular you can specify: the input folder for the dataset, the duration of the files to consider, the rateo between training and test set, the feature type, frame length, hop length, number of mel-bands and number of neighbors to consider during KNN. A figure of the confusion matrix and a text file containing all the evaluation metrics for each test will be saved inside the results folder.*

```terminal
python main_KNN.py --folder='..' --dur=.. --rateo=.. --feature_type='..' --frame_length=.. --hop_length=.. --n_mels=.. --n_neigh=..
```

### Multi KNN Classifier:
*Open the file 'run_KNN.sh' and uncomment and/or write your multiple test setup*

```terminal
. run_KNN.sh
```

### Single SVM Classifier:
*Run the training and then evaluate a model based on SVM and a certain feature to be specified. In particular you can specify: the input folder for the dataset, the duration of the files to consider, the rateo between training and test set, the feature type, frame length, hop length, number of mel-bands and number of max iteration and model order to consider during KNN. A figure of the confusion matrix and a text file containing all the evaluation metrics for each test will be saved inside the results folder.*

```terminal
python main_SVM.py --folder='..' --dur=.. --rateo=.. --feature_type='..' --frame_length=.. --hop_length=.. --n_mels=.. --order=.. --max_iter=..
```

### Multi SVM Classifier:
*Open the file 'run_SVM.sh' and uncomment and/or write your multiple test setup*

```terminal
. run_SVM.sh
```




