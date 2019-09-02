# RCAN Train
#python3 main.py --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --images_dir "/home/dongbin/DL/WDSR/datasets/Youku/Youku/train/HR" --outputs_dir "/home/dongbin/DL/RCAN/weights" --crop_size 224 --batch_size 16 --num_epochs 60 --lr 1e-4 --threads 8 --seed 123 --use_fast_loader

# ERCAN Train
#python3 main.py --scale 4 --images_dir "/home/dongbin/DL/WDSR/datasets/Youku/Youku/train/HR" --outputs_dir "/home/dongbin/DL/RCAN/weights" --crop_size 224 --batch_size 16 --num_epochs 60 --lr 1e-4 --threads 8 --seed 123

# RDN Train
python3 main.py --scale 4 --images_dir "/home/dongbin/DL/WDSR/datasets/Youku/Youku/train/HR" --outputs_dir "/home/dongbin/DL/RCAN/weights" --crop_size 224 --batch_size 2 --num_epochs 60 --lr 1e-4 --threads 8 --seed 123

# IDN Train
#python3 main.py --scale 4 --images_dir "/home/dongbin/DL/WDSR/datasets/Youku/Youku/train/HR" --outputs_dir "/home/dongbin/DL/RCAN/weights" --crop_size 224 --batch_size 8 --num_epochs 60 --lr 1e-4 --threads 8 --seed 123