# KNN 

dataset='Philharmonia-samples'

# direct signal with FIXED n_neigh VARIABLE feature params (same as the ones consider in viualization)
# feature_type="signal"
# python main_KNN.py --folder=$dataset --feature_type=$feature_type

# STFT with FIXED n_neigh VARIABLE feature params (same as the ones consider in viualization)
# feature_type="STFT"
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=128 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=128 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_neigh=5

# SC with FIXED n_neigh VARIABLE feature params (same as the ones consider in viualization)
# feature_type="SC"
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=128 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=512 --n_neigh=5

# MSTFT with FIXED n_neigh VARIABLE feature params (same as the ones consider in viualization)
# feature_type="MSTFT"
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_mels=8 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_mels=32 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=128 --n_mels=32 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=32 --n_neigh=5

# MFCC with FIXED n_neigh VARIABLE feature params (same as the ones consider in viualization)
# feature_type="MFCC"
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_mels=8 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_mels=32 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_mels=8 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=128 --n_mels=32 --n_neigh=5
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=32 --n_neigh=5

# 1) MSTFT with VARIABLE n_neigh FIXED feature params (best ones)
# feature_type="MSTFT"
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=32 --n_neigh=4
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=32 --n_neigh=8
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=32 --n_neigh=12
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=32 --n_neigh=16
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=32 --n_neigh=20

# 2) STFT with VARIABLE n_neigh FIXED feature params (best ones)
# feature_type="STFT"
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=128 --n_neigh=4
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=128 --n_neigh=8
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=128 --n_neigh=12
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=128 --n_neigh=16
# python main_KNN.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=128 --n_neigh=20
