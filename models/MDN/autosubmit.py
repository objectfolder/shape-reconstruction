import os
import sys
import time
with open('./sheet_remain.txt') as f:
    remain = f.readlines()
    remain_num = len(remain)
os.system('sbatch produce_sheets.submit')
while True:
    with open('./sheet_remain.txt') as f:
        remain = f.readlines()
    remain_num_new = len(remain)
    print("Remain number: ",remain_num_new)
    if remain_num_new<remain_num:
        remain_num = remain_num_new
        print("Submitting")
        os.system('sbatch produce_sheets.submit')
    time.sleep(5)