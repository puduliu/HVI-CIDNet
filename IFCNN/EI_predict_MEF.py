import os
import time

import torch

from model import myIFCNN

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image

from utils.myTransforms import denorm

fuse_scheme = 2
model_name = 'IFCNN-MEAN'

# load pretrained model
model = myIFCNN(fuse_scheme=fuse_scheme)
model.load_state_dict(torch.load('IFCNN-MEAN.pth'))
print("model state keys:", model.state_dict().keys())
model.eval()
model = model.cuda()

from utils.myDatasets import ImagePair

def get_image_paths(path1, path2):
    paths1 = sorted([os.path.join(path1, f) for f in os.listdir(path1)])
    paths2 = sorted([os.path.join(path2, f) for f in os.listdir(path2)])
    return paths1, paths2

datasets_num = 1
is_save = True  # if you do not want to save images, then change its value to False

begin_time = time.time()

method = 'IFCNN'
dataset = "MEF"
base_out = f'output/{dataset}'
base_in = f'/media/zyserver/data16t/lpd/HVI-CIDNet/datasets/inputs'

path_pairs = [
    (f"{base_out}/{method}/",
     f"{base_in}/over/",
     f"{base_in}/under/"), 
]

from pathlib import Path

dataset = "MEF"
is_gray = False  # Color (False) or Gray (True)
mean = [0.485, 0.456, 0.406]  # normalization parameters
std = [0.229, 0.224, 0.225]

for output_path, path1, path2 in path_pairs:
    os.makedirs(output_path, exist_ok=True)
    print("------------------------output_path = ", output_path)
    # print("----------------path1 = ", path1, "-----path1 = ", path2)
    # load source images
    img_paths1, img_paths2 = get_image_paths(path1, path2)
    for index, (img_path1, img_path2) in enumerate(zip(img_paths1, img_paths2)):# todo 遍历所有文件夹下面的数据
        print("----------------img_path1 = ", img_path1, "-----img_path2 = ", img_path2)
        file_name = Path(img_path1).name

        save_name = os.path.join(output_path, file_name)

        pair_loader = ImagePair(impath1=img_path1, impath2=img_path2,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)
                                ]))
        img1, img2 = pair_loader.get_pair() # tensor

        img1.unsqueeze_(0)
        img2.unsqueeze_(0)

        # perform image fusion
        with torch.no_grad():
            res = model(Variable(img1.cuda()), Variable(img2.cuda()))
            res = denorm(mean, std, res[0]).clamp(0, 1) * 255
            res_img = res.cpu().data.numpy().astype('uint8')
            img = res_img.transpose([1, 2, 0])

        # save fused images
        img = Image.fromarray(img)
        print("----------------------save_name = ", save_name)
        img.save(save_name)


proc_time = time.time() - begin_time
print('Total processing time of {} dataset: {:.3}s'.format(dataset, proc_time))
