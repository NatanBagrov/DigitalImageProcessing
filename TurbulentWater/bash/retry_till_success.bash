#!/bin/bash


time_to_sleep=60

while [[ true ]]
do
    echo "Running $@"
    eval "$@"

    if [[ "$?" = "0" ]]
    then
        break
    fi

    echo "Sleeping ${time_to_sleep} seconds"
    sleep ${time_to_sleep}
    time_to_sleep=$((time_to_sleep * 2))
done
