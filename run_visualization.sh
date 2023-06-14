# VISUALIZATION of a given sample

sample='Philharmonia-samples/trombone/trombone_A2_1_mezzo-forte_norma.wav'

python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=64 --n_mels=8
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=64 --n_mels=16
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=64 --n_mels=32
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=64 --n_mels=32
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=128 --n_mels=8
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=128 --n_mels=32


# sample='Philharmonia-samples/violin/violin_A3_1_fortissimo_arco-norma.wav'

# python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=64 --n_mels=8
# python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=64 --n_mels=16
# python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=64 --n_mels=32
# python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=64 --n_mels=32
# python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=128 --n_mels=8
# python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=128 --n_mels=32

## FOR NON MEL FEATURES

git branch -M mainsample='Philharmonia-samples/trombone/trombone_A2_1_mezzo-forte_norma.wav'

python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=64
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=128
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=256
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=128
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=256
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=512

sample='Philharmonia-samples/violin/violin_A3_1_fortissimo_arco-norma.wav'

python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=64
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=128
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=1024 --hop_length=256
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=128
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=256
python visualize_features.py --sample_file=$sample --dur=1 --frame_length=2048 --hop_length=512
