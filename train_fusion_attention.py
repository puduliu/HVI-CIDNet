import os
import torch
import random
from torchvision import transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from net.CIDNet_fusion_attention import CIDNet_DualHDR
from data.options import option
from measure import metrics
from eval import eval_dual
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime

opt = option().parse_args()

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    # print("---------------------------training")
    for batch in tqdm(training_data_loader):
        # print("---------------------------training_data_loader")
        im1, im2, im_gt, path1, path2 = batch[0], batch[1], batch[2], batch[3], batch[4]
        im1 = im1.cuda()
        im2 = im2.cuda()
        im_gt = im_gt.cuda()
        # print("---------------------------im1.shape = ", im1.shape)
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma, im2 ** gamma)  
        else:
            output_rgb = model(im1, im2)  
            
        gt_rgb = im_gt
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        iter += 1
        
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_print = loss_print + loss.item()
        loss_last_10 = loss_last_10 + loss.item()
        pic_cnt += 1
        pic_last_10 += 1
        if iter == train_len:
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                loss_last_10/pic_last_10, optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = transforms.ToPILImage()((output_rgb)[0].squeeze(0))
            gt_img = transforms.ToPILImage()((gt_rgb)[0].squeeze(0))
            if not os.path.exists(opt.val_folder+'training'):          
                os.mkdir(opt.val_folder+'training') 
            output_img.save(opt.val_folder+'training/test.png')
            gt_img.save(opt.val_folder+'training/gt.png')
    return loss_print, pic_cnt
                

def checkpoint(epoch):
    if not os.path.exists("./weights"):          
        os.mkdir("./weights") 
    if not os.path.exists("./weights/train"):          
        os.mkdir("./weights/train")  
    model_out_path = "./weights/train/epoch_fusion_att_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print('===> Loading datasets')
    if opt.SICE_mix or opt.SICE_grad or opt.SICE_test:
        if opt.SICE_mix:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_grad:
            train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
            
        if opt.SICE_test:
            train_set = get_SICE_fusion_training_set(opt.data_train_SICE,size=opt.cropSize)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
            test_set = get_SICE_fusion_eval_set(opt.data_val_SICE_test)
            testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

    else:
        raise Exception("should choose a dataset")
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = CIDNet_DualHDR().cuda()
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}_fusion.pth"
        model.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)      
    if opt.cos_restart_cyclic:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[(opt.nEpochs//4)-opt.warmup_epochs, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(optimizer=optimizer, periods=[opt.nEpochs//4, (opt.nEpochs*3)//4], restart_weights=[1,1],eta_mins=[0.0002,0.0000001])
    elif opt.cos_restart:
        if opt.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.warmup_epochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[opt.nEpochs - opt.start_epoch], restart_weights=[1],eta_min=1e-7)
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    
    # TODO 其实这几个损失函数没有毛病， 因为其它多曝光图像融合用的也是L1损失函数
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    D_loss = SSIM(weight=D_weight).cuda()
    E_loss = EdgeLoss(loss_weight=E_weight).cuda()
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').cuda()
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    print("--------------------------build_model start")
    model = build_model()
    print("--------------------------build_model over")
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    if not os.path.exists(opt.val_folder):          
        os.mkdir(opt.val_folder) 
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        # print("--------------------------epoch = ", epoch)
        epoch_loss, pic_num = train(epoch)
        # print("--------------------------train")
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            norm_size = True
            if opt.SICE_mix:
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.SICE_grad:
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
            if opt.SICE_test:
                output_folder = 'SICE_test_fusion_att/'
                label_dir = opt.data_valgt_SICE_test
                norm_size = False
            
            # im_dir = opt.val_folder + output_folder + '*.png' # TODO edit, 是不是要支持.jpg    
            im_dir = opt.val_folder + output_folder + '*.jpg' # TODO edit jpg, 推理出来好像是jpg格式
            eval_dual(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=opt.lol_v1, v2=opt.lolv2_real, alpha=0.8)
            
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False)
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)
        torch.cuda.empty_cache()
    
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("dataset: "+ output_folder + "\n")  
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"L1_weight: {opt.L1_weight}\n")  
        f.write(f"D_weight: {opt.D_weight}\n")  
        f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"P_weight: {opt.P_weight}\n")  
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        for i in range(len(psnr)):
            f.write(f"| {opt.start_epoch+(i+1)*opt.snapshots} | { psnr[i]:.4f} | {ssim[i]:.4f} | {lpips[i]:.4f} |\n")  
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
# ✅ 模型参数（state_dict）

# ❌ 优化器状态（optimizer.state_dict()）

# ❌ 学习率调度器状态（scheduler.state_dict()）

# ❌ 已训练的 epoch/step 信息（只靠 opt.start_epoch 人为指定）

# 所以目前的 恢复训练 其实更接近 “从某个 epoch 的模型参数开始再训练”，而不是严格意义上的 “resume training”。
        