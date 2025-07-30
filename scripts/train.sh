PORT=6666
GPU='0,1,2,3'
NUM_POINTS=121
IMG_SIZE=512
ENCODER='mit_b1' 
TRAINSET=YTBVOS2SEG_byvideo #'Your Train Dataset FileName, eg: YTBVOS2SEG_byvideo'
EVASL_VSOD='False'
EPOCH=2000
TRAIN_BATCH_SIZE=2
VAL_BATCH_SIZE=1
TRAIN_NUM_FRAME=5
THRESHOLD=0.5
TRAIN_ROOT='Your_train_data_path' 
VAL_ROOT='/Your_eval_data_path' 
SAVE_ROOT='./run_log' #'Your_running_log_save_path'
LOAD_FROM='None'
RESTORE_FROM='None'
INFER_SAVE=$SAVE_ROOT'/tmp_results'
SAVE_PATH=$SAVE_ROOT'/'$TRAINSET'/'
VAL_EVERY_EPOCH=1
NUM_SAVE_EPOCH=500
NUM_GPUS=`echo $GPU | awk -F ',' '{print NF}'`

#Model Stucture Config
ADD_MEM0='True'
ADD_MEM1='True'
OBJ_TO_PIXEL='True'
PIXEL_TO_OBJ='True'

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port $PORT tools/train.py \
  	--gpu $GPU \
  	--num_points $NUM_POINTS \
  	--img_size $IMG_SIZE \
	--threshold $THRESHOLD \
  	--epoch $EPOCH \
  	--train_batchsize $TRAIN_BATCH_SIZE \
  	--val_batchsize $VAL_BATCH_SIZE \
  	--train_root $TRAIN_ROOT \
	--val_root $VAL_ROOT \
	--train_num_frame $TRAIN_NUM_FRAME \
  	--train_dataset $TRAINSET \
	--train_dataset_repeat_list 3 \
	--val_dataset 'YTBOBJ2SEG_byvideo' 'DAVIS2SEG_byvideo' 'FBMS2SEG_byvideo'\
	--eval_vsod $EVASL_VSOD \
  	--infer_save $INFER_SAVE \
  	--save_path $SAVE_PATH \
  	--restore_from $RESTORE_FROM \
	--load_from $LOAD_FROM\
  	--encoder $ENCODER\
	--train_data_byvideo 'True' \
	--val_every_epoch $VAL_EVERY_EPOCH \
	--save_every_epoch $NUM_SAVE_EPOCH \
	--add_mem0 $ADD_MEM0 \
	--add_mem1 $ADD_MEM1 \
	--add_obj_to_pixel $OBJ_TO_PIXEL \
	--add_pixel_to_obj $PIXEL_TO_OBJ \
	--lr 6e-5

