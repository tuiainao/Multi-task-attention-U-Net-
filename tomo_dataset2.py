# dataloader
import torch
import torch.utils.data as data
import pdb
import os
import os.path
import numpy as np
import scipy.io as sio
#import matlab.engine
#eng = matlab.engine.start_matlab()
from IPython.core import debugger
debug = debugger.Pdb().set_trace
IMG_EXTENSIONS = ['.mat']
import torch.nn as nn



import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import pdb
from IPython.core import debugger
debug = debugger.Pdb().set_trace
import torch.nn.functional as F









class JointCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(*data)
        return data
debug
class JointRandomFlip(object):
    def __init__(self, rand=True):
        self.rand = rand

    def __call__(self, observe, geology,jiedian):
        random = torch.rand(1)
        if self.rand and random < 0.5:
            geology = geology.flip(2)   #flip?
            observe = observe.flip(2)
            jiedian = jiedian.flip(2)
        return observe, geology,jiedian


class JointNormalize(object):
    def __init__(self, norm=True):
        self.norm = norm

    def __call__(self, observe, geology,jiedian):
        if self.norm:
            #geology = (geology-1800.0)/2200.0
            #observe = 2*(observe+5.7)/(10.84+5.7)-1
            #geology = (geology-1.0)/299.0   ##1500 2500?
            observe = 2*(observe+470)/(718+470)-1    ## observe?
            jiedian = (jiedian-1)/(9-1)
            #observe[torch.abs(observe)>=0.01] = 0
            #observe = observe/observe.max()
        return observe, geology,jiedian    
        

def is_mat_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

"""
def mat_loader(path):
    try:
        data = sio.loadmat(path)             
    except Exception:
        print path   
        f = open('error.txt','a+');
        f.write(path);
        f.write("\n");
        f.close();
        num = int(path[-5]);
        listpath = list(path);
        listpath[-5] = str((num+1)%10);
        path = ''.join(listpath); 
        data = sio.loadmat(path)
        print('Error:', Exception)
        
    #finally:
    return data
"""    

def mat_loader(path):
    try:
        data = sio.loadmat(path,verify_compressed_data_integrity=False) 
        return data 
               
    except Exception:
        print (path)   
"""       f = open('error2.txt','a+');
       f.write(path);
       f.write("\n");
       f.close();
       num = int(path[-5]);
       listpath = list(path);
       listpath[-5] = str((num+1)%10);
       path = ''.join(listpath); 
       data = sio.loadmat(path)
       print('Error:', Exception)
"""       
   #finally:
    #return data


class TomoNet222(nn.Module):# model(observe, p=0.2, training=True)??model?TomoNet
    def __init__(self,x): 
        super(TomoNet222, self).__init__()
       
        
        self.interp = nn.Upsample(size=[5000, 199], mode='bilinear', align_corners=True).cuda()
       
        
        
        
    def forward(self, x, p=0.5, training=True): 
       
        x= self.interp(x).cuda()
        
        return  x




    
class DatasetFolder(data.Dataset):
    def __init__(self, root, flip=True, norm=True):
        #pdb.set_trace()
        self.flip = flip
        self.root = root
        self.norm = norm
        self.data = data
        self.obs_names = self.get_data(self.root+'/data')
        self.geo_names = self.get_con(self.obs_names)
        self.ind_names = self.get_con2(self.obs_names)
        
    def get_data(self, path):
        groups = [
            d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
        ]
        data_list = []
        for i in sorted(groups):
            group_path = os.path.join(path, i)
            if not os.path.isdir(group_path):
                continue
            data = [
                os.path.join(group_path, d)
                for d in os.listdir(group_path) if is_mat_file(d)
                ]
            data_list += data
        return data_list

    def get_con(self, data_list):
        con_list = [i.replace("data", "omodel") for i in data_list]
        return con_list
        debug
    def get_con2(self, data_list):
        con_list2 = [i.replace("data", "model") for i in data_list]
        return con_list2
        debug   
        
        
        
    def __getitem__(self, model):
        #pdb.set_trace()  
           
        observe_path = self.obs_names[model]
        geology_path = self.geo_names[model]
        jiedian_path = self.ind_names[model]
        
        geology = mat_loader(geology_path)['model']
        observe = mat_loader(observe_path)['E_obs']
        jiedian = mat_loader(jiedian_path)['index']
        
        observe = torch.from_numpy(observe)
        
        #observe = np.transpose(observe)
        #observe = nn.Upsample(size=[199, 5000], mode='bilinear', align_corners=True).cuda()
        
        observe = torch.unsqueeze(observe,0)
        geology = torch.from_numpy(geology)
        #geology = np.transpose(geology)
        jiedian = torch.from_numpy(jiedian)
        geology = geology.unsqueeze(0)  
        jiedian = jiedian.unsqueeze(0)
        
            
        transform = JointCompose([JointRandomFlip(self.flip), JointNormalize(self.norm)])
        observe, geology,jiedian = transform([observe, geology,jiedian])
        observe = observe[:,:,:]  ## observe?? shot , rec ,time  quanqu 
        observe = torch.unsqueeze(observe,0)
       
        #observe = nn.Upsample(size=[6200, 199], mode='bilinear', align_corners=True)(observe )
        observe =  observe.squeeze(0)
        #debug()
        
        return observe, geology,jiedian, observe_path, geology_path,jiedian_path

    def __len__(self):
        return len(self.geo_names)
