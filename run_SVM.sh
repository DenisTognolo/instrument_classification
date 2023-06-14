# SVM 

dataset='Philharmonia-samples'

# direct signal with FIXED order VARIABLE feature params (same as the ones consider in viualization)
# feature_type="signal"
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --max_iter=10000

# STFT with FIXED order VARIABLE feature params (same as the ones consider in viualization)
# feature_type="STFT"
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=128 --order=3
# # python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=128 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --order=3

# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=1536 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=4096 --hop_length=3072 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=8192 --hop_length=6144 --order=3

# SC with FIXED order VARIABLE feature params (same as the ones consider in viualization)
# feature_type="SC"
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --order=3
# # python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=128 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=512 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --order=3

# MSTFT with FIXED order VARIABLE feature params (same as the ones consider in viualization)
# feature_type="MSTFT"
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_mels=8 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_mels=32 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=128 --n_mels=32 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=32 --order=3

# MFCC with FIXED order VARIABLE feature params (same as the ones consider in viualization)
# feature_type="MFCC"
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_mels=8 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=64 --n_mels=32 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=128 --n_mels=32 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=32 --order=3

# ORDER TUNING
# feature_type="MSTFT"
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=128 --n_mels=32 --order=1
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=128 --n_mels=32 --order=3
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=128 --n_mels=32 --order=5
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=128 --n_mels=32 --order=7
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=128 --n_mels=32 --order=9
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=1024 --hop_length=128 --n_mels=32 --order=11

# MAX ITER TUNING
# feature_type="MSTFT"
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=8 --order=3 --max_iter=100
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=8 --order=3 --max_iter=1000
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=8 --order=3 --max_iter=10000
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --n_mels=8 --order=3 --max_iter=100000

# feature_type="STFT"
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --order=3 --max_iter=100
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --order=3 --max_iter=1000
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --order=3 --max_iter=10000
# python main_SVM.py --folder=$dataset --feature_type=$feature_type --frame_length=2048 --hop_length=256 --order=3 --max_iter=100000