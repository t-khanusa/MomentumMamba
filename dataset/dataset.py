import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import pandas as pd
import torch

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../../..")))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../..")))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

class RML2016_Dataset(Dataset):
    def __init__(self, 
                 hdf5_file,  
                 snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18], 
                 lengths = 256, 
                 modulations = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM'], 
                 one_hot = False,  
                 samples_per_key = None, 
                 ):
        self.file_path = hdf5_file
        self.modulations = modulations
        self.hot = one_hot
        self.sample_length = lengths
        self.data, self.length = [], []
        self.samples_per_key = int(samples_per_key) if samples_per_key else None
        self.modulation_map = dict(zip(modulations, range(len(modulations))))
        print("modulation map: {}".format(self.modulation_map))
        self.Xd = pickle.load(open(self.file_path, 'rb'), encoding='latin')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], self.Xd.keys())))), [1, 0])
        self.X, self.lbl = [], []
        for mod in mods:
            for snr in snrs:
                for i in range(self.Xd[(mod, snr)].shape[0] if not self.samples_per_key else self.samples_per_key):
                    self.X.append(self.Xd[(mod, snr)][i])
                for i in range(self.Xd[(mod, snr)].shape[0] if not self.samples_per_key else self.samples_per_key):  
                    self.lbl.append((self.modulation_map[mod], snr))
        self.length = len(self.X)
        self.Xd = []

    def DataDealing(self, sig):
        re, im = np.real(sig), np.imag(sig)
        return np.array([re, im])
    
    def one_hot(self, label, total):
        return np.array([1 if i==label else 0 for i in range(total)])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.X[index]
        label, snr = self.lbl[index]
        return data, label, snr
    
class AFOSR_Dataset(Dataset):
    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = np.array(target, dtype=np.int64)

    def __getitem__(self, index):
        # Read and process data
        data = pd.read_csv(self.data_path[index], header=None).values
        idx = np.linspace(0, len(data[:,0]) - 1, 256, dtype=int)
        samples = data[:, 1:][idx]
        samples = samples.transpose(1,0)
        
        # Add data normalization
        samples = (samples - np.mean(samples, axis=1, keepdims=True)) / (np.std(samples, axis=1, keepdims=True) + 1e-6)
        
        # Convert to tensor
        samples = torch.FloatTensor(samples)
        label = int(self.target[index])
        
            
        return samples, label

    def __len__(self):
        return len(self.data_path)


class UESTC_Dataset(Dataset):
    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = target
    

    def __getitem__(self, index):

        y = self.target[index]
        y = np.array(y, int)
        y = torch.from_numpy(y)

        samples = pd.read_csv(self.data_path[index], header=None).values

        a, b, c, d, e, f = np.array(samples[:,0], float)/16384, np.array(samples[:,1], float)/16384, np.array(samples[:,2], float)/16384, np.array(samples[:,3], float)/16.4, np.array(samples[:,4], float)/16.4, np.array(samples[:,5], float)/16.4
        idx = np.linspace(0, len(samples[:,0]) - 1, 512, dtype=int)
        ax, bx, cx, dx, ex, fx = a[idx], b[idx], c[idx], d[idx], e[idx], f[idx]
        samples = np.dstack((ax, bx, cx, dx, ex, fx))[0]

        # Add data normalization
        samples = (samples - np.mean(samples, axis=1, keepdims=True)) / (np.std(samples, axis=1, keepdims=True) + 1e-6)
        samples = samples.transpose(1, 0)

        samples = torch.from_numpy(samples).float()
        

        return samples, y

    def __len__(self):
        return len(self.data_path)
    

class MMAct_Dataset(Dataset):
    def __init__(self, data_path, target):
        self.data_path = data_path
        self.target = target

    def __getitem__(self, index):
        samples = pd.read_csv(self.data_path[index]).values
        idx = np.linspace(0, len(samples[:,0]) - 1, 256, dtype=int)
        ax, bx, cx, dx, ex, fx, gx, hx, ix, jx, kx, lx = samples[idx, 0], samples[idx, 1], samples[idx, 2], samples[idx, 3], samples[idx, 4], samples[idx, 5], samples[idx, 6], samples[idx, 7], samples[idx, 8], samples[idx, 9], samples[idx, 10], samples[idx, 11]
        samples = np.dstack((ax, bx, cx, dx, ex, fx, gx, hx, ix, jx, kx, lx))[0]
        samples = (samples - np.mean(samples, axis=1, keepdims=True)) / (np.std(samples, axis=1, keepdims=True) + 1e-6)
        samples = samples.transpose(1,0)    
        samples = torch.from_numpy(samples).float()
        label = int(self.target[index])
        return samples, label
    
    def __len__(self):
        return len(self.data_path)
    
    
    
    








        
# import torch, cv2
# from torch.utils.data import Dataset
# # from .preprocess import *
# import os
# import xlrd
# xlrd.xlsx.ensure_elementtree_imported(False, None)
# xlrd.xlsx.Element_has_iter = True
# import pandas
# import pdb
# import sys
# import numpy as np
# import cv2


def load_inertial(filename, startframe):
    if startframe < 0:
        startframe = 0
    df = pandas.read_csv(filename)
    df = df.iloc[2:, 1:]
    df = df.astype(float)
    # print(df)
    dfmean = df.mean(axis=0)
    # print(dfmean)
    df = df.div(dfmean, axis=1)
    df = df.to_numpy()
    # print(df.shape)
    acc = pow((pow(df[:, 0], 2) + pow(df[:, 1], 2) + pow(df[:, 2], 2)), 0.5)
    gyr = pow((pow(df[:, 3], 2) + pow(df[:, 4], 2) + pow(df[:, 5], 2)), 0.5)
    acc = np.asarray(acc)
    gyr = np.asarray(gyr)
    df = np.concatenate((df, acc[:, np.newaxis], gyr[:, np.newaxis]), axis=1)
    # print(df.shape)
    # print(df)
    df_normed = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))
    # print(df_normed)

    frames = df_normed[startframe:startframe + 150:3, :]
    # frames = frames[np.newaxis, :]
    # print(frames.shape)
    # frames = torch.from_numpy(frames).float()
    # print("########Inertial Load Done#######")
    return frames


class CMHADDataset(Dataset):
    """BBC Lip Reading dataset."""
    def build_file_list(self, dir, set):
        labels = ['Action1','Action2','Action3','Action4','Action5']
        completeList = []
        subject = 1
        while subject<=12:
            subdir = dir+"/Subject"+str(subject)
            LabelPath = xlrd.open_workbook(subdir+"/ActionOfInterestTVSubject"+str(subject)+".xlsx")
            sheet = LabelPath.sheet_by_index(0)
            min = 2
            max = 3
            for l in range(sheet.nrows-1):
                val = sheet.cell_value(l+1, 3)-sheet.cell_value(l+1, 2)
                if val < min:
                    min = val
                if val > max:
                    max = val
            # print("The Minimum action duration of this Subject is: "+str(min)+" seconds")
            # print("The Maximum action duration of this Subject is: "+str(max)+" seconds")
            valvideo = [10]
            # print("Validation Video include:", str(valvideo[0]))

            for m in range(sheet.nrows):
                if m==0:
                    continue
                Idirpath = subdir + "/InertialData/inertial_sub"+str(subject)+"_tv"+str(int(sheet.cell_value(m, 0)))+".csv"
                Vdirpath = subdir + "/VideoData/video_sub"+str(subject)+"_tv"+str(int(sheet.cell_value(m, 0)))+".avi"
                df = pandas.read_csv(Idirpath)
                MissFrames = 6005-len(df.index)
                midtime = sheet.cell_value(m, 3)/2 + sheet.cell_value(m, 2)/2
                if (set == "val") and (sheet.cell_value(m, 0) in valvideo) :
                    # print("Creating Vallidation dataset for Action"+ str(int(sheet.cell_value(m, 1))), "subject: ",str(subject))
                    startframe = int(50*midtime) - 75 #150 frames in total
                    endframe = int(50*midtime) + 74
                    startframe, endframe = self.check_overflow(startframe, endframe)

                    Vstartframe = int(15*midtime) - 22 #45 frames in total
                    Vendframe = int(15*midtime) + 22
                    Vstartframe, Vendframe = self.Vcheck_overflow(Vstartframe, Vendframe)

                    entry = (int(sheet.cell_value(m, 1)-1), Idirpath, startframe, startframe+149,MissFrames,Vdirpath,Vstartframe,Vstartframe+44,subject)
                    completeList.append(entry)

                elif (set == "train") and (sheet.cell_value(m, 0) not in valvideo) :
                    # print("Creating Training dataset for Action"+ str(int(sheet.cell_value(m, 1))), "subject: ",str(subject))
                    startframe = int(50*midtime) - 100 #200frames in total length, using only 150 frames from 200 as data augmentation
                    endframe = int(50*midtime) + 99
                    startframe, endframe = self.check_overflow(startframe, endframe)

                    Vstartframe = int(15*midtime) - 30 #45 frames in total
                    Vendframe = int(15*midtime) + 29
                    Vstartframe, Vendframe = self.Vcheck_overflow(Vstartframe, Vendframe)

                    for n in range(15):
                        entry = (int(sheet.cell_value(m, 1)-1), Idirpath, startframe+3*n, startframe+3*n+149,MissFrames,Vdirpath,Vstartframe+n,Vstartframe+n+44,subject)
                        completeList.append(entry)
            if set == "test":
                for o in valvideo:
                    startframe = MissFrames
                    midtime = (MissFrames+75)/50
                    Vstartframe = int(15*midtime) - 22
                    Idirpath = subdir + "/InertialData/inertial_sub"+str(subject)+"_tv"+str(o)+".csv"
                    Vdirpath = subdir + "/VideoData/video_sub"+str(subject)+"_tv"+str(o)+".avi"
                    # print("Creating Testing dataset for", "subject: ",str(subject))
                    while startframe <= 5851 and Vstartframe <= 1756:
                        entry = (0, Idirpath, startframe, startframe+149, MissFrames,Vdirpath, Vstartframe, Vstartframe+44,subject)
                        completeList.append(entry)
                        startframe = startframe + 10
                        Vstartframe = Vstartframe + 3
            subject = 1+subject
        print("Size of data : " + str(len(completeList)))
        # print(completeList)
        return labels, completeList

    def check_overflow(self, startframe, endframe):
        if startframe < 4: #avoid overflow
            endframe =  endframe+4-startframe
            startframe = 4
        elif endframe > 6000:
            startframe = startframe - (endframe-6000)
            endframe = 6000
        return startframe, endframe

    def Vcheck_overflow(self, startframe, endframe):
        if startframe < 0: #avoid overflow
            endframe = endframe - startframe
            startframe = 0
        elif endframe > 1800 :
            startframe = startframe - (endframe-1800)
            endframe = 1800
        return startframe, endframe

    def __init__(self, directory, set, augment=True):
        self.label_list, self.file_list = self.build_file_list(directory, set)
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        label, filename, startframe, endframe, MissFrames, Vdirpath, Vstartframe, Vendframe, subject = self.file_list[idx]

        IMU = load_inertial(filename, startframe - MissFrames)
        sample = IMU[:, :6]
        label = np.array(label, int)
        label = torch.from_numpy(label)

        # np.save(f"/content/drive/MyDrive/khanhnt/{label}.npy", sample)

        sample = sample.transpose(1, 0)
        sample = torch.from_numpy(sample).float()

        # vidframes = load_video(Vdirpath, Vstartframe)
        # temporalvolume = bbc(vidframes, self.augment)
        # sample = {'temporalvolume_x': Inerframes, 'temporalvolume_y': temporalvolume, 'label': torch.LongTensor([label]), 'MiddleTime':(startframe+75)/50, 'subject':subject}
        return sample, label


# CMHADDataset(directory="/hdd1/khanhnt/MAMC/TVGestureApplication", set="train", augment=True)
# CMHADDataset(directory="/hdd1/khanhnt/MAMC/TVGestureApplication", set="val", augment=True)
# CMHADDataset(directory="/hdd1/khanhnt/MAMC/TVGestureApplication", set="test", augment=True)