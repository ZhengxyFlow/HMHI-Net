# When Evaling with multiple GPUs, model needs return Tensors instead of list or dict.
GPU='0'
ENCODER='mit_b1'
IMG_SIZE=512
NUM_POINTS=121
INFER_DATASET_PATH='Your_eval_data_path'
INFER_DATASET='FBMS2SEG_byvideo' #'Your Dataset FileName, eg: FBMS'
INFER_MODEL_PATH='You Model Checkpoint Path'
INFER_SAVE='./output/'
VAL_BATCH_SIZE=1
EVASL_VSOD='False'

#Model Stucture Config
ADD_MEM0='True'
ADD_MEM1='True'
OBJ_TO_PIXEL='True'
PIXEL_TO_OBJ='True'

python tools/inference.py \
	--gpu $GPU \
	--encoder $ENCODER \
	--num_point $NUM_POINTS \
	--img_size $IMG_SIZE \
	--infer_dataset $INFER_DATASET\
	--infer_model_path $INFER_MODEL_PATH \
	--infer_save $INFER_SAVE \
	--val_batchsize $VAL_BATCH_SIZE \
	--infer_dataset_path $INFER_DATASET_PATH \
	--eval_vsod $EVASL_VSOD \
	--add_mem0 $ADD_MEM0 \
	--add_mem1 $ADD_MEM1 \
	--add_obj_to_pixel $OBJ_TO_PIXEL \
	--add_pixel_to_obj $PIXEL_TO_OBJ \