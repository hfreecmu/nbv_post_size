#!/bin/bash
# Written and maintained by Harry Freeman.  

# TODO add description


RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[36m'
NC='\033[0m' # No Color

echo -e "${GREEN}usage: ./full_extract_nbv_images.sh.sh"
echo -e "${GREEN}- Run from any path. Need to specify input and output directories"
echo -e "${GREEN}- TODO add some comment" 


###check args
NUM_ARGS="$#"

if [ $NUM_ARGS -ne 2 ]; then
    echo -e "${RED}Illegal number of parameters. Need exactly 2. Was $NUM_ARGS." >&2
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2

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

#make sure ros enabled
if command -v roscore &> /dev/null; then
    echo "ROS is sourced"
else
    echo -e "${RED}ROS is not sourced." >&2
    exit 1
fi

read -p "$(echo -e ${CYAN}Would you like to continue? Press Enter, Space, or Tab. $'\n> ')" -n 1 -r KEY
echo

for EXP_DIR in $INPUT_DIR/*/ ; do
    EXP_TYPE=$(basename "$EXP_DIR")

    RES_DIR=$OUTPUT_DIR/$EXP_TYPE

    [ -d $RES_DIR ] || mkdir $RES_DIR

    ./extract_nbv_images.sh $EXP_DIR $RES_DIR cluster

done

echo

echo -e "${GREEN}All done ...${NC}\n"