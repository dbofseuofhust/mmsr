# RCAN Test
#python3 example.py --scale 4 --num_rg 10 --num_rcab 20 --num_features 64 --weights_path "/home/dongbin/DL/RCAN/weights/RCAN_epoch_49.pth" --image_path "/deeplearning/Youku/Youku/test" --outputs_dir "/home/dongbin/DL/RCAN/data"

# ERCAN Test
#python3 example.py --scale 4 --weights_path "/home/dongbin/DL/RCAN/weights/ERCAN_epoch_1.pth" --image_path "/deeplearning/Youku/Youku/test" --outputs_dir "/home/dongbin/DL/RCAN/data"

# RDN Test
python3 example.py --scale 4 --weights_path "/home/dongbin/DL/RCAN/weights/RDN_epoch_15.pth" --image_path "/deeplearning/Youku/Youku/test" --outputs_dir "/home/dongbin/DL/RCAN/data"

# IDN Test
#python3 example.py --scale 4 --weights_path "/home/dongbin/DL/RCAN/weights/IDN_epoch_59.pth" --image_path "/deeplearning/Youku/Youku/test" --outputs_dir "/home/dongbin/DL/RCAN/data"

