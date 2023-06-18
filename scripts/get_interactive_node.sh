#!/bin/bash
srun --partition=viscam-interactive --nodelist=viscam1 --gres=gpu:1 --mem=30G --cpus-per-task=8 --pty bash 
