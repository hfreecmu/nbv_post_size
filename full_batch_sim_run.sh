#!/bin/bash
# Written and maintained by Harry Freeman.  

# assumes extract_nbv_images.sh has been called for all subdirs

# INPUT_DIR should be directory with all exp types in it
# OUTPUT_DIR should be directory with all exp types in it
# DUMMY_FILE is just a dummy file

#INPUT_DIR=INPUT_DIR
#OUTPUT_DIR=OUTPUT_DIR
#DUMMY_FILE=DUMMY_FILE
#./run_sizing_pipeline.sh $INPUT_DIR $OUTPUT_DIR $DUMMY_FILE

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[36m'
NC='\033[0m' # No Color


echo -e "${GREEN}usage: ./full_batch_sim_run.sh"
echo -e "${GREEN}- Run from any path. Need to specify data directory"
echo -e "${GREEN}- Calls sizing pipeline files for every bag in directory" 

###check args
NUM_ARGS="$#"

if [ $NUM_ARGS -ne 3 ]; then
    echo -e "${RED}Illegal number of parameters. Need exactly four. Was $NUM_ARGS." >&2
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
DUMMY_FILE=$3

if [ -d "${INPUT_DIR}" ] ; then
    echo -e "${CYAN}Input dir is: $INPUT_DIR"
else
    echo -e "${RED}Input dir is: $INPUT_DIR"
    echo -e "${RED}${INPUT_DIR} is not a directory." >&2
    exit 1
fi

if [ -d "${OUTPUT_DIR}" ] ; then
    echo -e "${CYAN}Input dir is: $OUTPUT_DIR"
else
    echo -e "${RED}Output dir is: $OUTPUT_DIR"
    echo -e "${RED}${OUTPUT_DIR} is not a directory." >&2
    exit 1
fi

if [ ! -f $DUMMY_FILE ]; then
    echo -e "${RED}${DUMMY_FILE} is not a valid dummy file." >&2
    exit 1
fi

read -p "$(echo -e ${CYAN}Would you like to continue? Press Enter, Space, or Tab. $'\n> ')" -n 1 -r KEY
echo

for EXP_DIR in $INPUT_DIR/*/ ; do
    EXP_TYPE=$(basename "$EXP_DIR")

    RES_DIR=$OUTPUT_DIR/$EXP_TYPE

    if [ ! -d $RES_DIR ]; then
        echo -e "${RED}${RES_DIR} does not exist." >&2
        continue
    fi

    RES_RES_DIR=$RES_DIR"_res"
    [ -d $RES_RES_DIR ] || mkdir $RES_RES_DIR

    python3 extract_target_centres.py --input_dir $EXP_DIR --output_dir $RES_DIR --bag_type cluster
    bash run_sizing_pipeline.sh $RES_DIR cluster $DUMMY_FILE $DUMMY_FILE
    python3 extract_gt_labels_sim.py --input_dir $EXP_DIR --output_dir $RES_DIR --res_dir $RES_RES_DIR

    python3 parse_results/parse_results.py --res_dir $RES_RES_DIR --res_type $EXP_TYPE --is_sim

done

echo

echo -e "${GREEN}All done ...${NC}\n"