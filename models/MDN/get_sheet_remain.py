import os
import os.path as osp
obj_cnt={}
for i in range(1,1001):
    obj_cnt[i]=0
    
sheet_location = '/viscam/projects/objectfolder_benchmark/benchmarks/Multi_Sensory_3D_Reconstruction/DATA_new/local_gt_sheets'
for root,dirs,files in os.walk(sheet_location):
    for file in files:
        file_path=osp.join(root,file)
        object_index, file_index = int(file_path.split('.')[0].split('/')[-2]), int(file_path.split('.')[0].split('/')[-1])
        obj_cnt[object_index]+=1
sheet_remain=[str(i)+'\n' for i in range(1,1001) if obj_cnt[i]<100]
print(sheet_remain)
with open("./sheet_remain.txt",'w') as f:
    f.writelines(sheet_remain)