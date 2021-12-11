import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import pdb
from IPython.core import debugger
from torch.autograd import Variable
debug = debugger.Pdb().set_trace
import torch.nn.functional as F
from torch.nn import Softmax

def TransformedDomainRecursiveFilter(Img, D, sigma):
    sqr2 = 1.4142135623730951
    a = torch.exp(torch.Tensor([-sqr2/sigma])).cuda()#???sigma
    V = torch.pow(a,D)
    b,c,h,w = Img.shape
    for i in range(1,w,1):
        Img[:,:,:,i] = Img[:,:,:,i] + V[:,:,:,i] * (Img[:,:,:,i-1] - Img[:,:,:,i])
    for i in range(w-2,-1,-1):
        Img[:,:,:,i] = Img[:,:,:,i] + V[:,:,:,i+1] * (Img[:,:,:,i+1] - Img[:,:,:,i])
    return Img



class Domain_Transform(nn.Module):
    def __init__(self, sigma_s = 60, sigma_r = 0.4, iter_num=2):
        super(Domain_Transform, self).__init__()
        #self.sigma_s = sigma_s
        #self.sigma_r = sigma_r
        self.sigma_s = nn.Parameter(torch.Tensor(1))
        self.sigma_r = nn.Parameter(torch.Tensor(1))
        self.sigma_s.data.fill_(sigma_s)
        self.sigma_r.data.fill_(sigma_r)
        self.iter_num = iter_num
    def forward(self, img, edge):      ###########forward?
        dHdx = (1 + self.sigma_s/self.sigma_r * edge)
        dVdy = dHdx.transpose(2,3)
        sqr3 = 1.7320508075688772
        for i in range(self.iter_num):
            sigma_i = self.sigma_s*sqr3*torch.pow(torch.Tensor([2]),self.iter_num-(i+1)).cuda()/torch.sqrt(torch.pow(torch.Tensor([4]),self.iter_num)-1).cuda();
            img = TransformedDomainRecursiveFilter(img, dHdx, sigma_i);
            img = img.transpose(2,3)
            img = TransformedDomainRecursiveFilter(img, dVdy, sigma_i);
            img = img.transpose(2,3)
        return img

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.n_channels = 1
        bilinear = False
        #bilinear = True
        self.inc = DoubleConv(32, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256,512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256,128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        #self.mux1 = CrissCrossAttention(64)
        #self.mux2 = CrissCrossAttention(256)
        #self.mux3 = CrissCrossAttention(256)
        #self.mux4 = CrissCrossAttention(512)

    def forward(self, x):
        x1 = self.inc(x)
        #x11=self.mux1(x1)
        #debug()
        x2 = self.down1(x1)
        #x22=self.mux2(x2)
        #x22=self.mux2(x2)

        x3 = self.down2(x2)
        #x33=self.mux3(x3)

        x4 = self.down3(x3)
        #x44=self.mux4(x4)

        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        #debug()
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size = [7,1], stride = [2,1], padding = [3,0]),
                                    nn.InstanceNorm2d(16),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(16, 16, kernel_size = [5,1], stride = [2,1], padding = [1,0]),
                                    nn.InstanceNorm2d(16),
                                    nn.ReLU(inplace = True),
                                   )
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size = [3,1], stride = [2,1], padding = [1,0]),
                                    nn.InstanceNorm2d(32),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(32, 32, kernel_size = [3,1], stride = [2,1], padding = [1,0]),
                                    nn.InstanceNorm2d(32),
                                    nn.ReLU(inplace = True),
                                    #nn.MaxPool2d(kernel_size = [2,2], stride = 2),
                                   )
        self.layer3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 64, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(inplace = True),
                                   ) 
        self.layer4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(128, 128, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(inplace = True),
                                   ) 
        self.layer5 = nn.Sequential(nn.Conv2d(128, 64, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(inplace = True),
                                    nn.Conv2d(64, 64, kernel_size = [3,3], stride = 1, padding = [1,1]),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(inplace = True),
                                   )
        self.layer6 = nn.Sequential(nn.Conv2d(512, 128, kernel_size = [13,4]),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace = True),
                                   )               
        self.interp = nn.Upsample(size=[64, 192], mode='bilinear', align_corners=True).cuda()
        self.conv_1x1_output = nn.Conv2d(1 * 2, 1, 1, 1)                             
                                                                                                       
    def forward(self, x):
        #b,c,h,w = x.shape
        #x = x.view(b*c,-1,h,w)
       
        
        out = self.layer1(x)
        #outout = self.interp(out).cuda()
        
        out = self.layer2(out)
        #debug()
        #out = self.layer3(out)
        #debug()
        #out = self.layer4(out)
        #out = self.layer5(out)
        
        out = self.interp(out).cuda()
        debug()
        #out = self.conv_1x1_output(torch.cat([out, outout], dim=1))
        
        #out = out.view(b,-1)
        #debug()             
        return out



                
class TomoNet(nn.Module):# model(observe, p=0.2, training=True)??model?TomoNet
    def __init__(self): 
        super(TomoNet, self).__init__()
        self.mmunet = UNet()
        #self.mmunet_ge = Dense_Unet_GE()
        self.mmout_seg = nn.Conv2d(64, 3, kernel_size=1, stride=1)
        self.mmout_inv = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        
        #self.decoder = Dense_Unet(in_chan = 1,out_chan = 7,num_conv = 4,filters = 64).cuda()
        self.encoder = Encoder()
       # self.lianjie = lianjie()
       # self.juanji = juanji()
        
        #self.interp = nn.Upsample(size=[400, 400], mode='bilinear', align_corners=True).cuda()
        self.interp1 = nn.Upsample(size=[70, 200], mode='bilinear', align_corners=True).cuda()
        
        
        
    def forward(self, x, p=0.5, training=True): 
    
        x=self.encoder(x).cuda()
        #7*32*64*192
        #x= self.interp(x).cuda()
        
        out = self.mmunet(x)
        #debug()
        out = self.interp1(out).cuda()
        out_t_inv = self.mmout_inv(out)              # bz * 5 * 240 * 240
        out_t_seg = self.mmout_seg(out) 
        out_t_inv = (self.sigmoid(out_t_inv))
        
        #debug()
        #debug()
        #out1 = self.interp1(out1).cuda()

        
        #out1= torch.nn.functional.adaptive_avg_pool2d(out1, (1,1)).cuda()
        #out1=self.juanji(out1).cuda()
      
        
        #out1=out1.view(out1.size(0),-1)
        #out1=self.lianjie(out1)
        
        #debug()
        return  out_t_seg, out_t_inv
         
