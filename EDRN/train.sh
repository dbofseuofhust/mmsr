#python3 main.py --scale 4 --images_dir "/home/dongbin/DL/WDSR/datasets/Youku/Youku/train/HR" --outputs_dir "/home/dongbin/DL/EDRN/weights" --crop_size 224 --batch_size 16 --num_epochs 60 --lr 1e-4 --threads 8 --seed 123
#python3 inference.py --scale 2 --n_resblocks 32 --n_feats 256 --res_scale 0.1 --weights_path r'/home/dongbin/EDSR-PyTorch/model_pytorch/EDSR_x2.pt'

#===================================================================== AWSRN =======================================================================
#-------------AWSRNx4 train/test
#python main.py --model AWSRN --n_resblocks 4 --scale 4 --images_dir "/home/dongbin/DL/WDSR/datasets/Youku/Youku/train/HR" --outputs_dir "/home/dongbin/DL/EDRN/weights" --num_epochs 60 --crop_size 224 --batch_size 16 --lr 1e-4 --threads 8 --seed 123 --use_fast_loader
#python main.py --model AWSRN  --n_resblocks 4 --scale 4  --pre_train ../experiment/AWSRNx4/model/model_latest.pt --save AWSRNx4 --test_only  --dir_data ../../DATA/ --data_test Set5

#===================================================================== AWSRN-M =======================================================================
#-------------AWSRN_Mx4 train test
#python main.py --model AWSRN --n_resblocks 3 --scale 4  --save AWSRN_Mx4  --epochs 1000 --reset --patch_size 192
#python main.py --model AWSRN  --n_resblocks 3 --scale 4  --pre_train ../experiment/AWSRN_Mx4/model/model_latest.pt --save AWSRN_Mx4 --test_only  --dir_data ../../DATA/ --data_test Set5

#===================================================================== AWSRN-S =======================================================================
#-------------AWSRN_Sx4 train test
#python main.py --model AWSRN --n_resblocks 1 --scale 4 --save AWSRN_Sx4  --epochs 1000  --reset --patch_size 192
#python main.py --model AWSRN --n_resblocks 1 --scale 4  --pre_train ../experiment/AWSRN_Sx4/model/model_latest.pt --save AWSRN_Sx4 --test_only  --dir_data ../../DATA/ --data_test Set5

#===================================================================== AWSRN-SD =======================================================================
#-------------AWSRN_SDx4 train test
#python main.py --model AWSRND --n_resblocks 1 --n_feats 16 --block_feats 128 --scale 4  --save AWSRN_SDx4  --epochs 1000 --reset --patch_size 192
#python main.py --model AWSRND  --n_resblocks 1 --n_feats 16 --block_feats 128 --scale 4  --pre_train ../experiment/AWSRN_SDx4/model/model_latest.pt --save AWSRN_SDx4 --test_only  --dir_data ../../DATA/ --data_test Set5

#===================================================================== DRCA-x4 =======================================================================
python3 main.py --scale 4 --n_resgroups 5 --n_resblocks 36 --n_feats 64 --images_dir "/home/dongbin/DL/WDSR/datasets/Youku/Youku/train/HR" --outputs_dir "/home/dongbin/DL/EDRN/weights" --crop_size 224 --batch_size 1 --num_epochs 60 --lr 1e-4 --threads 8 --seed 123
#python3 inference.py --scale 4 --n_resgroups 5 --n_resblocks 36 --n_feats 64 --weights_path r'/home/dongbin/EDSR-PyTorch/model_pytorch/EDSR_x2.pt'