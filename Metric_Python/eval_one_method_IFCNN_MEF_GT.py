import numpy as np
from PIL import Image
from Metric_GT import *
from natsort import natsorted
from tqdm import tqdm
import os
import statistics
import warnings
from skimage.metrics import structural_similarity as ssim  # 导入计算 SSIM 的函数
from skimage.metrics import peak_signal_noise_ratio as psnr
warnings.filterwarnings("ignore")

def preprocess_image(img):
    return img.astype(np.float32) / 255.0

def calculate_psnr_skimage(img1, gt): # TODO channel_axis= -1 计算的是彩色图
    psnr1 = psnr(img1, gt, data_range=255)
    return psnr1

def calculate_ssim_average(img1, gt): # TODO channel_axis= -1 计算的是彩色图
    """计算图像的平均 SSIM"""
    # TODO skimage 会自动在每个通道上分别计算 SSIM，再取平均
    # ssim1 = ssim(img1, gt, channel_axis=-1, data_range=1) # RGB图像计算
    # ssim2 = ssim(img2, gt, channel_axis=-1, data_range=1)

    # TODO 默认按灰度图处理，只看二维结构（没有通道信息）
    ssim1 = ssim(img1, gt, data_range=1)
    return ssim1


def evaluation_one(gt_name, f_name):
    f_img = Image.open(f_name).convert('L')
    gt_img = Image.open(gt_name).convert('L')

    f_img_int = np.array(f_img).astype(np.int32)
    f_img_double = np.array(f_img).astype(np.float32)

    gt_img_int = np.array(gt_img).astype(np.int32)
    gt_img_double = np.array(gt_img).astype(np.float32)

    EN = EN_function(f_img_int)
    MI = MI_function(gt_img_int, f_img_int, gray_level=256)

    SF = SF_function(f_img_double)
    SD = SD_function(f_img_double)
    AG = AG_function(f_img_double)
    # PSNR = PSNR_function(gt_img_double, f_img_double)
    MSE = MSE_function(gt_img_double, f_img_double)
    VIF = VIF_function(gt_img_double, f_img_double)
    CC = CC_function(gt_img_double, f_img_double)
    # SCD = SCD_function(gt_img_double, f_img_double)
    # Qabf = Qabf_function(gt_img_double, f_img_double)
    # Nabf = Nabf_function(gt_img_double, f_img_double)
    # SSIM = SSIM_function(gt_img_double, f_img_double)
    # MS_SSIM = MS_SSIM_function(gt_img_double, f_img_double)
    
    f_img_double = preprocess_image(f_img_double)
    gt_img_double = preprocess_image(gt_img_double)
    PSNR = calculate_psnr_skimage(gt_img_double, f_img_double)
    SSIM = calculate_ssim_average(gt_img_double, f_img_double)
    return EN, MI, SF, AG, SD, CC, VIF, MSE, PSNR, SSIM

if __name__ == '__main__':
    with_mean = True
    EN_list = []
    MI_list = []
    SF_list = []
    AG_list = []
    SD_list = []
    CC_list = []
    SCD_list = []
    VIF_list = []
    MSE_list = []
    PSNR_list = []
    Qabf_list = []
    Nabf_list = []
    SSIM_list = []
    MS_SSIM_list = []
    filename_list = ['']

    dataset_name = 'MEF'
    gt_dir = "/media/zyserver/data16t/lpd/HVI-CIDNet/datasets/gt"
    Method = 'IFCNN'
    f_dir = "/media/zyserver/data16t/lpd/HVI-CIDNet/IFCNN/output/MEF/IFCNN"
    save_dir = '../Metric'
    os.makedirs(save_dir, exist_ok=True)
    metric_save_name = os.path.join(save_dir, 'metric_{}_{}.xlsx'.format(dataset_name, Method))
    filelist1 = natsorted(os.listdir(gt_dir))
    filelist2 = natsorted(os.listdir(f_dir))

    # 检查两个列表的长度是否相同
    if len(filelist1) != len(filelist2):
        raise ValueError("ir_dir 和 vi_dir 中的文件数量不匹配")

    eval_bar = tqdm(zip(filelist1, filelist2))  # 同时遍历两个文件列表

    for item1, item2 in eval_bar:
        gt_name = os.path.join(gt_dir, item1)
        f_name = os.path.join(f_dir, item2)
        # EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(gt_name, f_name)
        EN, MI, SF, AG, SD, CC, VIF, MSE, PSNR, SSIM = evaluation_one(gt_name, f_name)
        EN_list.append(EN)
        MI_list.append(MI)
        SF_list.append(SF)
        AG_list.append(AG)
        SD_list.append(SD)
        CC_list.append(CC)
        VIF_list.append(VIF)
        MSE_list.append(MSE)
        PSNR_list.append(PSNR)
        # Qabf_list.append(Qabf)
        # Nabf_list.append(Nabf)
        SSIM_list.append(SSIM)
        # MS_SSIM_list.append(MS_SSIM)
        filename_list.append(item2)
        eval_bar.set_description("{} | {}".format(Method, item2))
    if with_mean:
    # 添加均值
        EN_list.append(np.mean(EN_list))
        MI_list.append(np.mean(MI_list))
        SF_list.append(np.mean(SF_list))
        AG_list.append(np.mean(AG_list))
        SD_list.append(np.mean(SD_list))
        CC_list.append(np.mean(CC_list))
        SCD_list.append(np.mean(SCD_list))
        VIF_list.append(np.mean(VIF_list))
        MSE_list.append(np.mean(MSE_list))
        PSNR_list.append(np.mean(PSNR_list))
        Qabf_list.append(np.mean(Qabf_list))
        Nabf_list.append(np.mean(Nabf_list))
        SSIM_list.append(np.mean(SSIM_list))
        MS_SSIM_list.append(np.mean(MS_SSIM_list))
        filename_list.append('mean')

        # TODO edit start
        EN_mean = np.mean(EN_list)
        MI_mean = np.mean(MI_list)
        SF_mean = np.mean(SF_list)
        AG_mean = np.mean(AG_list)
        SD_mean = np.mean(SD_list)
        CC_mean = np.mean(CC_list)
        SCD_mean = np.mean(SCD_list)
        VIF_mean = np.mean(VIF_list)
        MSE_mean = np.mean(MSE_list)
        PSNR_mean = np.mean(PSNR_list)
        Qabf_mean = np.mean(Qabf_list)
        Nabf_mean = np.mean(Nabf_list)
        SSIM_mean = np.mean(SSIM_list)
        MS_SSIM_mean = np.mean(MS_SSIM_list)
        # 写入txt文件，路径和文件名自行调整
        txt_path = os.path.join(save_dir, f'metric_mean_{dataset_name}_{Method}_GT.txt')
        with open(txt_path, 'a') as f:
            f.write(f"Metric mean results for {dataset_name} - {Method}\n")
            f.write("="*40 + "\n")
            f.write(f"EN: {EN_mean:.4f}\n")
            f.write(f"MI: {MI_mean:.4f}\n")
            f.write(f"SF: {SF_mean:.4f}\n")
            f.write(f"AG: {AG_mean:.4f}\n")
            f.write(f"SD: {SD_mean:.4f}\n")
            f.write(f"CC: {CC_mean:.4f}\n")
            f.write(f"SCD: {SCD_mean:.4f}\n")
            f.write(f"VIF: {VIF_mean:.4f}\n")
            f.write(f"MSE: {MSE_mean:.4f}\n")
            f.write(f"PSNR: {PSNR_mean:.4f}\n")
            f.write(f"Qabf: {Qabf_mean:.4f}\n")
            f.write(f"Nabf: {Nabf_mean:.4f}\n")
            f.write(f"SSIM: {SSIM_mean:.4f}\n")
            f.write(f"MS_SSIM: {MS_SSIM_mean:.4f}\n")
        # TODO edit end
        
        ## 添加标准差
        EN_list.append(np.std(EN_list))
        MI_list.append(np.std(MI_list))
        SF_list.append(np.std(SF_list))
        AG_list.append(np.std(AG_list))
        SD_list.append(np.std(SD_list))
        CC_list.append(np.std(CC_list[:-1]))
        SCD_list.append(np.std(SCD_list))
        VIF_list.append(np.std(VIF_list))
        MSE_list.append(np.std(MSE_list))
        PSNR_list.append(np.std(PSNR_list))
        Qabf_list.append(np.std(Qabf_list))
        Nabf_list.append(np.std(Nabf_list))
        SSIM_list.append(np.std(SSIM_list))
        MS_SSIM_list.append(np.std(MS_SSIM_list))
        filename_list.append('std')

    ## 保留三位小数
    EN_list = [round(x, 3) for x in EN_list]
    MI_list = [round(x, 3) for x in MI_list]
    SF_list = [round(x, 3) for x in SF_list]
    AG_list = [round(x, 3) for x in AG_list]
    SD_list = [round(x, 3) for x in SD_list]
    CC_list = [round(x, 3) for x in CC_list]
    SCD_list = [round(x, 3) for x in SCD_list]
    VIF_list = [round(x, 3) for x in VIF_list]
    MSE_list = [round(x, 3) for x in MSE_list]
    PSNR_list = [round(x, 3) for x in PSNR_list]
    Qabf_list = [round(x, 3) for x in Qabf_list]
    Nabf_list = [round(x, 3) for x in Nabf_list]
    SSIM_list = [round(x, 3) for x in SSIM_list]
    MS_SSIM_list = [round(x, 3) for x in MS_SSIM_list]

    EN_list.insert(0, '{}'.format(Method))
    MI_list.insert(0, '{}'.format(Method))
    SF_list.insert(0, '{}'.format(Method))
    AG_list.insert(0, '{}'.format(Method))
    SD_list.insert(0, '{}'.format(Method))
    CC_list.insert(0, '{}'.format(Method))
    SCD_list.insert(0, '{}'.format(Method))
    VIF_list.insert(0, '{}'.format(Method))
    MSE_list.insert(0, '{}'.format(Method))
    PSNR_list.insert(0, '{}'.format(Method))
    Qabf_list.insert(0, '{}'.format(Method))
    Nabf_list.insert(0, '{}'.format(Method))
    SSIM_list.insert(0, '{}'.format(Method))
    MS_SSIM_list.insert(0, '{}'.format(Method))