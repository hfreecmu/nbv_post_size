#!/bin/bash
# Written and maintained by Harry Freeman.  

# Not sure if it will conflict if run in conda env so maybe better if not

# ./extract_nbv_images.sh $INPUT_DIR $OUTPUT_DIR tsdfroi
# copy outputted files to a text file after complete

#INPUT_DIR=PATH_TO_INPUT_DIR
#OUTPUT_DIR=PATH_TO_OUTPUT_DIR
#BAG_TYPE=TYPE_OF_BAG
#./extract_nbv_images.sh $INPUT_DIR $OUTPUT_DIR $BAG_TYPE

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[36m'
NC='\033[0m' # No Color

VALID_BAG_TYPES=("tsdfroi" "dissim" "linear", "cluster")
ROSBAG_RATE=5
KILL_LIMIT=10

echo -e "${GREEN}usage: ./extract_nbv_images.sh"
echo -e "${GREEN}- Run from any path. Need to specify input and output directories"
echo -e "${GREEN}- Calls 1_extract_images.py for every bag in directory" 

###check args
NUM_ARGS="$#"

if [ $NUM_ARGS -ne 3 ]; then
    echo -e "${RED}Illegal number of parameters. Need exactly three. Was $NUM_ARGS." >&2
    exit 1
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
BAG_TYPE=$3

if [ -d "${INPUT_DIR}" ] ; then
    echo -e "${CYAN}Input dir is: $INPUT_DIR"
else
    echo -e "${RED}Input dir is: $INPUT_DIR"
    echo -e "${RED}${INPUT_DIR} is not a directory." >&2
    exit 1
fi

if [ -d "${OUTPUT_DIR}" ] ; then
    echo -e "${CYAN}Output dir is: $OUTPUT_DIR"
else
    echo -e "${RED}Output dir is: $OUTPUT_DIR"
    echo -e "${RED}${OUTPUT_DIR} is not a directory." >&2
    exit 1
fi

if [[ "${VALID_BAG_TYPES[*]}" =~ "$BAG_TYPE" ]]; then
    echo -e "${CYAN}Bag type is: $BAG_TYPE"
else
    echo -e "${RED}Bag type is: $BAG_TYPE"
    echo -e "${RED}${BAG_TYPE} is not in ${VALID_BAG_TYPES[*]}." >&2
    exit 1
fi

###

#make sure ros enabled
if command -v roscore &> /dev/null; then
    echo "ROS is sourced"
else
    echo -e "${RED}ROS is not sourced." >&2
    exit 1
fi

# read -p "$(echo -e ${CYAN}Would you like to continue? Press Enter, Space, or Tab. $'\n> ')" -n 1 -r KEY
# echo

# if [[ $KEY != "" ]]; then 
#     echo -e "${RED}Cancelling." >&2
#     exit 1
# fi

FAILED_LIST=()
FULL_LIST=()
for SUB_DIR in $INPUT_DIR/*/ ; do
    BASE_NAME=$(basename "$SUB_DIR")

    if [[ "$BASE_NAME" != *"$BAG_TYPE"* ]]; then
        continue
    fi

    echo -e "${GREEN}Processing $BASE_NAME"
    FULL_LIST+=("$BASE_NAME;")

    BAG_FILE="${SUB_DIR}/run.bag"
    
    if [ ! -f $BAG_FILE ]; then
        FAILED_LIST+=("$BASE_NAME;")
        echo -e "${RED}No bag file for ${BASE_NAME}. Skipping." >&2
        continue
    fi

    BAG_OUT_DIR="${OUTPUT_DIR}/${BASE_NAME}"
    [ -d $BAG_OUT_DIR ] || mkdir $BAG_OUT_DIR

    #start roscore in background
    roscore &
    sleep 2

    #set use_sim_time
    rosparam set use_sim_time true

    #call extract images.py
    #WARNING WARNING WARNING
    #change name of this script need to change kill all command
    if [ "$BAG_TYPE" = "cluster" ]; then 
        python2 1_extract_images.py --data_dir ${BAG_OUT_DIR} --is_sim & 
    else
        python2 1_extract_images.py --data_dir ${BAG_OUT_DIR} --process_rect_images --process_joints & 
    fi;
    PYTHON_PID=$!
    sleep 5

    #play rosbag
    rosbag play -r ${ROSBAG_RATE} --clock ${BAG_FILE}

    #clean up

    #start with python
    #kill the command. if takes to long, kill -9
    kill $PYTHON_PID
    KILL_COUNTER=0
    while kill -0 $PYTHON_PID; do 
        sleep 1
        if [ "$KILL_COUNTER" -lt "$KILL_LIMIT" ]; then
            ((KILL_COUNTER++))
        else
            echo -e "${RED}Python taking too long to shut down. Killing" >&2
            kill -9 $PYTHON_PID
            FAILED_LIST+=("$BASE_NAME;")
            echo -e "${RED}Adding ${BASE_NAME} to failed lists." >&2
            break
        fi
    done


    rosnode kill --all

    #incase any lingering
	PYTHON_PIDS=$(pgrep -f "1_extract_images.py")
    if [ -n "$PYTHON_PIDS" ]; then
        kill -9 $PYTHON_PIDS
    fi
    
    killall -9 rosmaster
	killall -9 roscore

    sleep 2
    echo -e "${GREEN}Done Processing $BASE_NAME${NC}"
done
 
echo

if [ ${#FAILED_LIST[@]} -gt 0 ]; then
  echo -e "${RED}Failed lists are:"
  echo -e "${RED}${FAILED_LIST[*]}"
fi

echo -e "${GREEN}Full lists are:"
echo -e "${GREEN}${FULL_LIST[*]}"

echo -e "${GREEN}All done ...${NC}\n"