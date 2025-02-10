import os 

path_to_reid_ws = '/home/dongwooklee1201/morin/Research/Masters_Dissertation/mpfan/mpfan-marine'
path_to_data = '/media/dongwooklee1201/Data_Storage_A1/Datasets'   #to make symbolic link to data work
memory = '724g'
cpus = 10
gpus = 'all'
port = 14000
image = 'daldidan/mpfan:latest'
name = 'mpfan-marine-test'

command = "docker run -v {}:{} -v {}:{} --memory {} --shm-size=8g --cpus={} --gpus {} -p {}:{} --name {} --rm -it {}".format(
    path_to_reid_ws,"/mpfan/",path_to_data,"/mpfan/Datasets/",memory,cpus,gpus,port,port,name,image
)

print("################################################")
print('[run_docker.py executing] ',command)
print("################################################")

os.system(command)

# CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=localhost torchpack dist-run -v -np 1 python tools/train.py configs_reid/reid_nuscenes_pts/training/training_pointnet.py --seed 66  --run-dir runs/