# LF-SloMo
Angularly consistent light field video interpolation. 
  
This is the code that was used for scene flow estimation from light fields in:
P. David, M. Le Pendu, C. Guillemot, "Angularly Consistent Light Field Video Interpolation." In Proceedings of the IEEE International Conference on Multimedia and Expo. 2020. 

Author: Pierre David. 

It is largely inspired from the PyTorch implementation of "Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation" by Avinash Paliwal (https://github.com/avinashpaliwal/Super-SloMo). Every original file that was not heavily modified is marked with the following copyright: "#Copyright (c) 2018 Avinash Paliwal". 

## Testing:

You need to have estimated the bidirectional optical flow between the two frames you want to interpolate.
In our article, we used the methods proposed in Pierre David, Mikaël Le Pendu, Christine Guillemot, "Scene Flow Estimation from Sparse Light Fields Using a Local 4D Affine Model." In IEEE Transactions on Computational Imaging. 2020. 

Once you have the optical flows saved as .flo, the input files should be sorted as such:  

|---Input Folder  
	|  
	|  
	|----im0  
	|     |----frame 1 (FRAME001.png)  
	|     |----frame 2 (FRAME002.png)  
	|     ⋮  
	|     ⋮  
	|     |----frame N (FRAMENNN.png)  
	|  
	|  
	|----im1  
	|     |----frame 2 (FRAME001.png)  
	|     |----frame 3 (FRAME002.png)  
	|     ⋮  
	|     ⋮  
	|     |----frame N+1 (FRAMENNN.png)  
	|  
	|  
	|----f01  
	|     |----flow 1->2 (FRAME001.flo)  
	|     |----flow 2->3 (FRAME002.flo)  
	|     ⋮  
	|     ⋮  
	|     |----flow N->N+1 (FRAMENNN.flo)  
	|  
	|  
	|----f10  
	      |----flow 2->1 (FRAME001.flo)  
	      |----flow 3->2 (FRAME002.flo)  
	           ⋮  
	           ⋮  
	      |----flow N+1->N (FRAMENNN.flo)  

You can then launched:  
> python3 image_to_slomo.py --input ${PATH_TO_INPUT_FOLDER} --checkpoint ${PATH_TO_MODEL} --output ${PATH_TO_OUTPUT_FOLDER}

Additional argument:  
"--sf": Specify scale to be applied to frame rate. Default: 2  
"--batch_size": Specify batch size for faster conversion. This will depend on your cpu/gpu memory. Default: 1  

Python Version: 3.6.5  
PyTorch Version: 1.2  

## Training:

> python3 train.py --dataset_root ${PATH_TO_DATASET} --checkpoint_dir ${OUTPUT_PATH_FOR_MODEL}

Required arguments:   
"--dataset_root": path to the training folder, ordered as mentioned in the testing part. The training folder should also include the ground truth intermediate frames in a folder named "imt" and the pre-computed intermediate flows t->1 in a folder named "ft1"  
"--checkpoint_dir": path to folder for saving checkpoints   

Optional arguments:  
"--checkpoint": path of checkpoint for pretrained model  
"--train_continue": If resuming from checkpoint, set to True and set `checkpoint` path. Default: False  
"--epochs": number of epochs to train. Default: 146  
"--train_batch_size": batch size for training. Default: 6  
"--validation_batch_size": batch size for validation. Default: 10  
"--init_learning_rate": set initial learning rate. Default: 0.0001  
"--milestones": Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [73, 110]  
"--progress_iter": frequency of reporting progress and validation. N: after every N iterations. Default: 100  
"--checkpoint_epoch": checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB. Default: 5  

## Evaluating:

In order to computed the different metrics used in our paper (namely PSNR, SSIM and Light Field Epipolar Consistency) described in the article, please use the following functions in "evaluate_LF_interp.py":

evaluate_method_sintel(root, root_gt, root_sf) for the whole Sintel Light Field Video dataset  
compute_LFinterp_metrics(LF, LF_gt, disp, vh_ratio=1) for any light field

## Citation:

@inproceedings{david2020angularly,  
  title={Angularly Consistent Light Field Video Interpolation},  
  author={David, Pierre and Le Pendu, Mika{\"e}l and Guillemot, Christine},  
  booktitle={2020 IEEE International Conference on Multimedia and Expo (ICME)},  
  pages={1--6},  
  year={2020},  
  organization={IEEE}  
}