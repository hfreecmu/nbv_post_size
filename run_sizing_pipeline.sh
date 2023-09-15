#!/bin/bash
# Written and maintained by Harry Freeman.  

# assumes extract_nbv_images.sh has been called
# assumes extract_target_centres.py has been called

# DATA_DIR should be OUTPUT_DIR from two above commands

#DATA_DIR=DATA_DIR
#BAG_TYPE=TYPE_OF_BAG
#RAFT_CKPT=/home/frc-ag-3/harry_ws/viewpoint_planning/segment_exp/src/fruitlet_disparity/models/iraftstereo_rvc.pth
#SEG_MODEL_PATH=/home/frc-ag-3/harry_ws/fruitlet_2023/labelling/segmentation/turk/mask_rcnn/mask_best.pth
#./run_sizing_pipeline.sh $DATA_DIR $BAG_TYPE $RAFT_CKPT $SEG_MODEL_PATH

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[36m'
NC='\033[0m' # No Color

VALID_BAG_TYPES=("tsdfroi" "dissim" "linear")
KILL_LIMIT=10

echo -e "${GREEN}usage: ./run_sizing_pipeline.sh"
echo -e "${GREEN}- Run from any path. Need to specify data directory"
echo -e "${GREEN}- Calls sizing pipeline files for every bag in directory" 

###check args
NUM_ARGS="$#"

if [ $NUM_ARGS -ne 4 ]; then
    echo -e "${RED}Illegal number of parameters. Need exactly four. Was $NUM_ARGS." >&2
    exit 1
fi

DATA_DIR=$1
BAG_TYPE=$2
RAFT_CKPT=$3
SEG_MODEL_PATH=$4

if [ -d "${DATA_DIR}" ] ; then
    echo -e "${CYAN}Input dir is: $DATA_DIR"
else
    echo -e "${RED}Input dir is: $DATA_DIR"
    echo -e "${RED}${DATA_DIR} is not a directory." >&2
    exit 1
fi

if [[ "${VALID_BAG_TYPES[*]}" =~ "$BAG_TYPE" ]]; then
    echo -e "${CYAN}Bag type is: $BAG_TYPE"
else
    echo -e "${RED}Bag type is: $BAG_TYPE"
    echo -e "${RED}${BAG_TYPE} is not in ${VALID_BAG_TYPES[*]}." >&2
    exit 1
fi

if [ ! -f $RAFT_CKPT ]; then
    echo -e "${RED}${RAFT_CKPT} is not a valid raft file." >&2
    exit 1
fi

if [ ! -f $SEG_MODEL_PATH ]; then
    echo -e "${RED}${SEG_MODEL_PATH} is not a valid seg file." >&2
    exit 1
fi

read -p "$(echo -e ${CYAN}Would you like to continue? Press Enter, Space, or Tab. $'\n> ')" -n 1 -r KEY
echo

if [[ $KEY != "" ]]; then 
    echo -e "${RED}Cancelling." >&2
    exit 1
fi

for SUB_DIR in $DATA_DIR/*/ ; do
    BASE_NAME=$(basename "$SUB_DIR")

    if [[ "$BASE_NAME" != *"$BAG_TYPE"* ]]; then
        continue
    fi

    echo -e "${GREEN}Processing $BASE_NAME"

    #don't need 1 as did already

    #step 2 get _image idss
    # echo -e "${GREEN}Getting image ids for $BASE_NAME"
    # python3 2_get_image_ids.py --data_dir $SUB_DIR

    # #step 3 extract disparities
    # echo -e "${GREEN}Extracting disparities for $BASE_NAME"
    # python3 3_extract_disparities.py --data_dir $SUB_DIR --raft_restore_ckpt $RAFT_CKPT

    # #step 4 extract point clouds
    # echo -e "${GREEN}Extracting point clouds for $BASE_NAME"
    # python3 4_extract_point_clouds.py --data_dir=$SUB_DIR

    # #step 5 segment
    # echo -e "${GREEN}Segmenting for $BASE_NAME"
    # python3 5_segment.py --data_dir $SUB_DIR --model_path $SEG_MODEL_PATH

    #don't need 6 as did already

    # #step 7 filter segments
    # echo -e "${GREEN}Filtering segments for $BASE_NAME"
    # python3 7_filter_segments.py --data_dir $SUB_DIR

    # #step 8 associate
    # echo -e "${GREEN}Associating for $BASE_NAME"
    # python3 8_associate.py --data_dir $SUB_DIR

    # #step 9 fit ellipses
    # echo -e "${GREEN}Fitting ellipses for $BASE_NAME"
    # python3 9_fit_ellipse.py --data_dir $SUB_DIR

    #step 10 size
    echo -e "${GREEN}Sizing for $BASE_NAME"
    python3 10_size.py --data_dir $SUB_DIR

done

echo

echo -e "${GREEN}All done ...${NC}\n"