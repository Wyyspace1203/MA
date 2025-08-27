seed=2019
bs=96
nav_t=1
gpu_id=0
epoch=100
 python image_target.py --cls_par 0.0 --da uda --dset ntupsb --s 0 --output_src ckm1_bt64/source/ --output ckm1_bt64/target_bt32/loss1/ --batch_size 32
