import warnings
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as L
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import dataset as data
import numpy as np
import random
import argparse
from torch.utils.data import ConcatDataset
import cv2
from mopoevae import MoPoEVAE
import torch
import os
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_num_threads(16)

def train(opt):

    # enable deterministic behavior if flag is set
    if opt.det:
        print("Deterministic mode enabled!")
        S = 3407  # https://arxiv.org/abs/2109.08203 :)
        random.seed(S)
        torch.manual_seed(S)
        torch.cuda.manual_seed(S)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        L.seed_everything(S)

    # select device
    device = torch.device(
        "cuda:"+opt.device if torch.cuda.is_available() and opt.device != 'cpu' else "cpu")

    # performs a 8:1:1 train/val/test split
    def train_val_test_dataset(dataset):
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, shuffle=False, random_state=42)
        val_idx, test_idx = train_test_split(val_idx, test_size=0.5, shuffle=False, random_state=42)
        return Subset(dataset, train_idx), Subset(dataset, val_idx),Subset(dataset, test_idx)
    
    frequence_L = int(np.ceil(np.log(opt.ws)/np.log(2)))

    # create dataloaders
    print("Creating dataloaders...")
    j3_train = data.WificamDataset(opt.data+"j3/"+str(opt.imgsize)+"/csi.csv",augPath='', windowSize=opt.ws, frequence_L=frequence_L,random_sample=opt.random,temporal_encoding=opt.tenc)
    j3_val = data.WificamDataset(opt.data+"j3/"+str(opt.imgsize)+"/csi.csv",augPath='', windowSize=opt.ws, frequence_L=frequence_L,random_sample=False, temporal_encoding=opt.tenc)
    j3_test = data.WificamDataset(opt.data+"j3/"+str(opt.imgsize)+"/csi.csv",augPath='', windowSize=opt.ws, frequence_L=frequence_L,random_sample=False, temporal_encoding=opt.tenc)
    datasetTrainJ3, _, _ = train_val_test_dataset(j3_train)
    _, datasetValJ3, _ = train_val_test_dataset(j3_val)
    _,_, datasetTestJ3 = train_val_test_dataset(j3_test)

    if not opt.test:
        dataloader_train = DataLoader(ConcatDataset([datasetTrainJ3]), batch_size=opt.bs, shuffle=True, num_workers=opt.workers, drop_last=True)
        dataloader_val = DataLoader(ConcatDataset([datasetValJ3]), batch_size=opt.bs*8, shuffle=False, num_workers=opt.workers, drop_last=True)
        model = MoPoEVAE(weight_ll=True, lr=opt.lr,sequence_length=opt.ws, z_dim=opt.zdim, frequence_L= frequence_L,aggregate_method=opt.am,imgMean=j3_train.imgMean,imgStd=j3_train.imgStd,log=opt.log)

        # copy python files to run folder
        if not os.path.exists("runs/"+opt.name):
            os.makedirs("runs/"+opt.name)
        os.system("cp *.py runs/"+opt.name+"/")

        print("Training...")
        wandb_logger = WandbLogger(log_model=False, project="EnterYourWandbProjectName", entity="EnterYourWandbEntity", config=opt, name=opt.name) if opt.log else None
        callbacks = [ModelCheckpoint(monitor='val_loss', mode='min', save_last=False, filename='bestLoss', dirpath='runs/'+opt.name),
                    ModelCheckpoint(monitor='val_kl', mode='min', save_last=False,filename='bestKl', dirpath='runs/'+opt.name),
                    ModelCheckpoint(monitor='val_ll', mode='max', save_last=False,filename='bestLl', dirpath='runs/'+opt.name),
                    ModelCheckpoint(monitor='FID', mode='min', save_last=False,filename='bestFID', dirpath='runs/'+opt.name),
                    EarlyStopping(monitor="val_loss", mode='min', patience=25, min_delta=10.0, verbose=True)]

        trainer = L.Trainer(accelerator='gpu', devices=[device.index], gradient_clip_val=1.0, logger= wandb_logger, callbacks=callbacks, max_epochs=opt.epochs)
        trainer.fit(model, dataloader_train, dataloader_val)
    else:
        print("Reconstructing...")
        dataloader_test = DataLoader(datasetTestJ3, batch_size=opt.bs*8,shuffle=False, num_workers=opt.workers, drop_last=True)
        model = MoPoEVAE.load_from_checkpoint(f'runs/{opt.name}/bestLoss.ckpt', weight_ll=True, lr=opt.lr, sequence_length=opt.ws, z_dim=opt.zdim,frequence_L= frequence_L,aggregate_method=opt.am, map_location=device,imgMean=j3_train.imgMean,imgStd=j3_train.imgStd,log=opt.log)
        model.to(device)
        model.eval()

        # Metrics
        FID = FrechetInceptionDistance().to(device)
        KID = KernelInceptionDistance(subset_size=100, subsets=1000).to(device)
        SSIM = np.zeros((len(dataloader_test.dataset)))
        RMSE = np.zeros((len(dataloader_test.dataset)))
        PSNR = np.zeros((len(dataloader_test.dataset)))
        ssim = StructuralSimilarityIndexMeasure()

        # get means and stds from model
        imgMean = model.imgMean.reshape(1, 3, 1, 1).cpu().numpy()
        imgStd = model.imgStd.reshape(1, 3, 1, 1).cpu().numpy()

        idx = 0
        batchCount = 1
        for batch in tqdm(dataloader_test):
            spectrogram, image = batch
            spectogram_time, spectogram = spectrogram
            image_time, image = image
            spectrogram = spectogram_time.to(device), spectogram.to(device)
            image = image_time.to(device), image.to(device)

            with torch.no_grad():
                reconstruction = model.decode(model.encode_subset([spectrogram, image], [0]))[1][0][1]
            image = image[1].permute(0, 2, 3, 1).cpu().numpy()
            reconstruction = reconstruction.permute(0, 2, 3, 1).cpu().numpy()

            # slow and ugly but works :)
            for i in range(len(reconstruction)):
                data_sample = image[i][..., ::-1]  # Convert to BGR
                pred_sample = reconstruction[i][..., ::-1]  # Convert to BGR

                #channel first 
                data_sample = data_sample.transpose(2,0,1)
                pred_sample = pred_sample.transpose(2,0,1)
    
                # reverse normalization
                data_sample = data_sample * imgStd + imgMean
                pred_sample = pred_sample * imgStd + imgMean

                # clip to [0,1]
                data_sample = np.clip(data_sample, 0, 1)
                pred_sample = np.clip(pred_sample, 0, 1)

                # convert to uint8
                data_sample = (data_sample*255).astype(np.uint8)
                pred_sample = (pred_sample*255).astype(np.uint8)

                # compute metrics
                FID.update(torch.tensor(pred_sample,dtype=torch.uint8).to(device), real=True)
                FID.update(torch.tensor(data_sample,dtype=torch.uint8).to(device), real=False)
                KID.update(torch.tensor(pred_sample,dtype=torch.uint8).to(device), real=True)
                KID.update(torch.tensor(data_sample,dtype=torch.uint8).to(device), real=False)  
                SSIM[idx] = ssim(torch.tensor(pred_sample,dtype=torch.float), torch.tensor(data_sample,dtype=torch.float))
                RMSE[idx] = np.sqrt(np.mean((data_sample - pred_sample)**2))
                # compute PSNR
                PSNR[idx] = cv2.PSNR(data_sample, pred_sample)

                # drop batch dimension
                data_sample = data_sample[0]
                pred_sample = pred_sample[0]

                # channel last
                data_sample = data_sample.transpose(1,2,0)
                pred_sample = pred_sample.transpose(1,2,0)
                img = np.concatenate((data_sample, pred_sample), axis=1)
                # check if out dir exists
                if not os.path.exists("runs/"+opt.name+"/out"):
                    os.makedirs("runs/"+opt.name+"/out")
                    os.makedirs("runs/"+opt.name+"/out/combined")
                    os.makedirs("runs/"+opt.name+"/out/real")
                    os.makedirs("runs/"+opt.name+"/out/fake")
                cv2.imwrite("runs/"+opt.name+"/out/combined/"+str(idx)+".png",img)
                cv2.imwrite("runs/"+opt.name+"/out/real/"+str(idx)+".png",data_sample) 
                cv2.imwrite("runs/"+opt.name+"/out/fake/"+str(idx)+".png",pred_sample) 
                idx += 1
            batchCount += 1

        fid = FID.compute().item()
        kid,std = KID.compute()
        ssim_score = np.mean(SSIM)
        rmse = np.mean(RMSE)
        psnr = np.mean(PSNR)

        # print results
        print("FID: %.4f" % fid, "KID: %.4f" % kid, "+- %.4f" % std, "SSIM: %.4f" % ssim_score, "RMSE: %.4f" % rmse, "PSNR: %.4f" % psnr)

        # write metrics to file in run folder
        with open("runs/"+opt.name+"/metrics.txt", "w") as f:
            f.write("FID: "+str(fid)+"\n")
            f.write("KID: "+str(kid.item())+" "+str(std.item())+"\n")
            f.write("SSIM: "+str(ssim_score)+"\n")
            f.write("RMSE: "+str(rmse)+"\n")
            f.write("PSNR: "+str(psnr)+"\n")

        # save SSIM, RMSE and PSNR time series
        np.savetxt("runs/"+opt.name+"/SSIM.txt", SSIM, fmt="%.4f")
        np.savetxt("runs/"+opt.name+"/RMSE.txt", RMSE, fmt="%.4f")
        np.savetxt("runs/"+opt.name+"/PSNR.txt", PSNR, fmt="%.4f")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3,help='optimizer learning rate')
    parser.add_argument('--epochs', type=int, default=100,help='number of epochs')
    parser.add_argument('--imgsize', type=int, default=640,help='image size (320 or 640)')
    parser.add_argument('--zdim', type=int, default=128,help='latent dimension')
    parser.add_argument('--bs', type=int, default=32,help='total batch size for all GPUs')
    parser.add_argument('--ws', type=int, default=151,help='spectrogram window size (number of WiFi packets)')
    parser.add_argument('--workers', type=int, default=8,help='maximum number of dataloader workers')
    parser.add_argument('--device', default='0',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--name', default='default', help='wandb run name')
    parser.add_argument('--data', default='data/wificam/', help='data directory')
    parser.add_argument('--augment', default='', type=str, metavar='PATH', help='path to augmentation parameters (default: none)')
    parser.add_argument('--test', action='store_true',help='perform reconstruction and evaluation only')
    parser.add_argument('--am', default='concat',choices=['uniform', 'gaussian','concat'], type=str,help='type of aggregation method')
    parser.add_argument('--random', action='store_true', help='use random image sampling within CSI window')
    parser.add_argument('--tenc', action='store_true',help='use temporal encoding')
    parser.add_argument('--det', action='store_true', help='enable deterministic behavior')
    parser.add_argument('--log', action='store_true', help='enable wandb logging')
    opt = parser.parse_args()
    print(opt)

    train(opt)
    torch.cuda.empty_cache()

    # training command:
    # python3 train.py --name mopoevae_ct --data data/wificam/ --epochs 50 --am concat --tenc --device 0

    # testing command:
    # python3 train.py --name mopoevae_ct --data data/wificam/ --epochs 50 --am concat --tenc --device 0 --test

    # create video from images:
    # ffmpeg -framerate 100 -i %d.png -c:v libx264 -pix_fmt yuv420p demo.mp4



