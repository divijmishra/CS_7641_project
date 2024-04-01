#!/usr/bin/bash

# After loading onto the server with VS Code SSH, run the following commands:

# To use a different GPU: change "V100-32GB" to the appropriate name as per
# https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096 
# To change the time required, change t01:00:00 to tHH:MM:SS

gpu=V100-32GB
hours=4

while getopts ":g:h:" opt; do
    case "${opt}" in
        g) gpu=$OPTARG;;
        h) hours=$OPTARG;;
    esac
done

srun --partition=coc-gpu --gres=gpu:1 -C $gpu -N 1 -t $hours:00:00 --pty $SHELL
