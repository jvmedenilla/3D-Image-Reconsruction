'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
from torch.autograd import Variable


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False) #THE FIRST NUM (3) IS THE NUMBER OF INPUT CHANNELS
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        return p2


def FPN101():
    # return FPN(Bottleneck, [2,4,23,3])
    return FPN(Bottleneck, [2,2,2,2])
    
class FRGB(nn.Module):
    def __init__(self):
        super(FRGB, self).__init__()
        self.conv1 = nn.Conv2d(256,128,3,padding=1)
        self.gn1 = nn.GroupNorm(32,128)
        self.ReLU1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128,128,3,padding=1)
        self.gn2 = nn.GroupNorm(32,128)
        self.ReLU2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128,128,1)
        self.gn3 = nn.GroupNorm(32,128)
    
    def forward(self,x):
        x = self.ReLU1(self.gn1(self.conv1(x)))
        x = self.ReLU2(self.gn2(self.conv2(x)))
        x = self.gn3(self.conv3(x))
        return x

class Fspatial(nn.Module):
    def __init__(self):
        super(Fspatial, self).__init__()
        self.conv1 = nn.Conv2d(256,128,3,padding=1)
        self.gn1 = nn.GroupNorm(32,128)
        self.ReLU1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128,128,3,padding=1)
        self.gn2 = nn.GroupNorm(32,128)
        self.ReLU2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128,1,1)
    
    def forward(self,x):
        x = self.ReLU1(self.gn1(self.conv1(x)))
        x = self.ReLU2(self.gn2(self.conv2(x)))
        x = self.conv3(x)
        return x
    
class TwoLayerCNN(nn.Module):
    def __init__(self):
        super(TwoLayerCNN, self).__init__()
        self.conv1 = nn.Conv2d(64,128,1)
        self.gn1 = nn.GroupNorm(32,128)
        self.ReLU1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128,128,1)
        self.gn2 = nn.GroupNorm(32,128)
    
    
    def forward(self,x):
        x = self.ReLU1(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return x

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, op_true, op_pred, mask, N):
        
        ####################
        num = (2*mask*op_true*op_pred).sum(dim = [2,3])
        d1 = (mask*op_true).sum(dim=[2,3])
        d2 = (mask*op_pred).sum(dim=[2,3])
        s = (num/(d1+d2)).sum()
        
        ####################
        #Should be equivalent to this:
#         for i in range(N):
#             num = 2*torch.sum(mask*op_true[i]*op_pred[i])
#             denom = torch.sum(mask*op_true[i])+torch.sum(mask*op_pred[i])
#             s = s + num/denom
    
        return s/N
    

    
class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss,self).__init__()
    
    def forward(self, op_true, op_pred, mask, N):
        
        num = mask*((op_true*(torch.log(op_pred))) + (1-op_true)*torch.log(1-op_pred))                 
        s = num.sum()
        s = s/(N*mask.sum()) 
    
        return s
        
        
        

class ComboNet(nn.Module):
    def __init__(self, batchSize, numPeChannels, learningRate):
        super(ComboNet, self).__init__()
        self.f_FPN = FPN(Bottleneck, [3,4,6,3])
        self.f_RGB = FRGB()
        self.f_depth = TwoLayerCNN()
        self.f_spatial = Fspatial()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
        self.lossBCE = BCELoss()
        self.lossDICE = DiceLoss()
        
        #Make denominator tensor for positional encoding (don't want to run duplicate work)
        denom = torch.zeros(batchSize,numPeChannels,256,256)
        idx = torch.ones(256,256)
        for i in range(numPeChannels):
            denom[:,i,:,:] = 200**(2*i*idx/numPeChannels)
        self.denom = denom
        self.numPeChannels = numPeChannels
        
        
    def forward(self,x1,x2,z):
        st = time.time()
        #RGB feature processing
        x1 = self.f_FPN(x1)
        x1_lowres = self.f_RGB(x1)
        x1 = F.interpolate(x1_lowres,scale_factor=2, mode='bilinear') #Upsampling step: convert 128x128 to 256x256 img
        
        #Depth feature processing
        x2 = self.positionEncoding(x2,z)
        x2_lowres = F.interpolate(x2,scale_factor=0.5, mode='bilinear') #Downsampled depth difference image
        x2 = self.f_depth(x2)
        x2_lowres = self.f_depth(x2_lowres)
        
        #Combine features and pass them through final CNN
        x = torch.cat((x1,x2), 1) #second arg specifies which dimension to concatenate on, we want channel dimension which is 1
        x = self.f_spatial(x)
        
        #Get low res OPlane for loss computation, use inner product (eqn 14 from paper)   
        multp  = x1_lowres * x2_lowres 
        x_lowres = multp.sum(dim = 1, keepdim = True)
        
        #Normalize both outputs so all values are between 0 and 1
        x = x - x.min()
        x = x/x.max()
        x_lowres = x_lowres - x_lowres.min()
        x_lowres = x_lowres/x_lowres.max()
        
        return x, x_lowres
    
    def positionEncoding(self, depth, z):
        """
        Computes the positional encoding (as defined by the paper) for a depth
        - depth: the input depth image
        - z: the distance we wish to evaluate
        """
        
        depth = F.interpolate(depth,scale_factor=0.5, mode='bilinear')
        s = depth.size()
        pe = torch.zeros(s[0],self.numPeChannels,s[2],s[3])
        num = z-depth
        pe[:,0::2,:,:] = torch.sin(50*num/self.denom[:,0::2,:])
        pe[:,1::2,:,:] = torch.cos(50*num/self.denom[:,1::2,:])
        
        return pe
    
    def step(self,x_RGB,x_depth,mask,z_vals,op_true_highres):
        """
        Iterates over a single training step, ie one image with a set of N values in the range [z_min, z_max]
        - x: input batch
        - y: expected labels for batch
        """
        self.optimizer.zero_grad() #Reset parameter gradients to 0
        
        #Get the outputs for each values of z
        N = z_vals.size()[0]
        op_highres = torch.zeros(N,1,256,256)
        op_lowres = torch.zeros(N,1,128,128)
        
        st = time.time()
        for i, z in enumerate(z_vals):
            op_highres_i, op_lowres_i = self.forward(x_RGB,x_depth,z)
            op_highres[i,:,:,:] = op_highres_i
            op_lowres[i,:,:,:] = op_lowres_i
        print("Forward time: ",round(time.time()-st,2))
            
        
        #Calculate the loss based on the predicted OPlanes for all z values
        lambda_BCE, lambda_DICE = 1,1
        mask_lowres = F.interpolate(mask,scale_factor=0.25, mode='bilinear')
        mask = F.interpolate(mask,scale_factor=0.5, mode='bilinear')
        op_true_highres = F.interpolate(op_true_highres,scale_factor=0.5, mode='bilinear')
        op_true_lowres = F.interpolate(op_true_highres,scale_factor=0.5, mode='bilinear')
        
        loss_highres = lambda_BCE*self.lossBCE(op_true_highres, op_highres, mask, N) + lambda_DICE*self.lossDICE(op_true_highres, op_highres, mask, N)
        loss_lowres = lambda_BCE*self.lossBCE(op_true_lowres, op_lowres, mask_lowres, N) + lambda_DICE*self.lossDICE(op_true_lowres, op_lowres, mask_lowres, N)    

        loss = loss_highres + loss_lowres
        st = time.time()
        loss.backward()
        print("Backward time: ",round(time.time()-st,2))
        
        self.optimizer.step()

        return loss.detach().cpu().numpy()