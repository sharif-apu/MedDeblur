#wrappers for convenience
import torch.nn as nn
from torch.nn.init import xavier_normal_ , kaiming_normal_
import copy
from functools import partial
import torch
import torch.nn.functional as F 

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv9x9(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4, bias=True)

def get_weight_init_fn( activation_fn  ):
    """get weight_initialization function according to activation_fn
    Notes
    -------------------------------------
    if activation_fn requires arguments, use partial() to wrap activation_fn
    """
    fn = activation_fn
    if hasattr( activation_fn , 'func' ):
        fn = activation_fn.func

    if  fn == nn.LeakyReLU:
        negative_slope = 0 
        if hasattr( activation_fn , 'keywords'):
            if activation_fn.keywords.get('negative_slope') is not None:
                negative_slope = activation_fn.keywords['negative_slope']
        if hasattr( activation_fn , 'args'):
            if len( activation_fn.args) > 0 :
                negative_slope = activation_fn.args[0]
        return partial( kaiming_normal_ ,  a = negative_slope )
    elif fn == nn.ReLU or fn == nn.PReLU :
        return partial( kaiming_normal_ , a = 0 )
    else:
        return xavier_normal_
    return

def conv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 , activation_fn= None , use_batchnorm = False , pre_activation = False , bias = True , weight_init_fn = None ):
    """pytorch torch.nn.Conv2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        conv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    conv = nn.Conv2d( in_channels , out_channels , kernel_size , stride , padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    try:
        weight_init_fn( conv.weight )
    except:
        print( conv.weight )
    layers.append( conv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

def deconv( in_channels , out_channels , kernel_size , stride = 1  , padding  = 0 ,  output_padding = 0 , activation_fn = None ,   use_batchnorm = False , pre_activation = False , bias= True , weight_init_fn = None ):
    """pytorch torch.nn.ConvTranspose2d wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        deconv(3,32,3,1,1,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))

    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    deconv = nn.ConvTranspose2d( in_channels , out_channels , kernel_size , stride ,  padding , output_padding , bias = bias )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    weight_init_fn( deconv.weight )
    layers.append( deconv )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

def linear( in_channels , out_channels , activation_fn = None , use_batchnorm = False ,pre_activation = False , bias = True ,weight_init_fn = None):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    Arguments:
        activation_fn : use partial() to wrap activation_fn if any argument is needed
        weight_init_fn : a init function, use partial() to wrap the init function if any argument is needed. default None, if None, auto choose init function according to activation_fn 

    examples:
        linear(3,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 ))
    """
    if not pre_activation and use_batchnorm:
        assert not bias

    layers = []
    if pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( in_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    linear = nn.Linear( in_channels , out_channels )
    if weight_init_fn is None:
        weight_init_fn = get_weight_init_fn( activation_fn )
    weight_init_fn( linear.weight )

    layers.append( linear )
    if not pre_activation :
        if use_batchnorm:
            layers.append( nn.BatchNorm2d( out_channels ) )
        if activation_fn is not None:
            layers.append( activation_fn() )
    return nn.Sequential( *layers )

class BasicBlock(nn.Module):
    """pytorch torch.nn.Linear wrapper
    Notes
    ---------------------------------------------------------------------
    use partial() to wrap activation_fn if arguments are needed 
    examples:
        BasicBlock(32,32,activation_fn = partial( torch.nn.LeakyReLU , negative_slope = 0.1 , inplace = True ))
    """
    def __init__(self, in_channels , out_channels , kernel_size , stride = 1 , use_batchnorm = False , activation_fn = partial( nn.ReLU ,  inplace=True ) , last_activation_fn = partial( nn.ReLU , inplace=True ) , pre_activation = False , scaling_factor = 1.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv( in_channels , out_channels , kernel_size , stride , kernel_size//2 ,  activation_fn , use_batchnorm )
        self.conv2 = conv( out_channels , out_channels , kernel_size , 1 , kernel_size//2 , None , use_batchnorm  , weight_init_fn = get_weight_init_fn(last_activation_fn) )
        self.downsample = None
        if stride != 1 or in_channels != out_channels :
            self.downsample = conv( in_channels , out_channels , 1 , stride , 0 , None , use_batchnorm )
        if last_activation_fn is not None:
            self.last_activation = last_activation_fn()
        else:
            self.last_activation = None
        self.scaling_factor = scaling_factor
    def forward(self , x ):
        residual = x 
        if self.downsample is not None:
            residual = self.downsample( residual )

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual * self.scaling_factor
        if self.last_activation is not None:
            out = self.last_activation( out )

        return out


class MultiKernelConv2(nn.Module):
    def __init__(self, cin, cout, stride=1, padding_mode='zeros'):
        super(MultiKernelConv2, self).__init__()

        self.square_conv = nn.Conv2d(in_channels=cin, out_channels=cout,
                                     kernel_size=(9, 9), stride=stride,
                                     padding=4, bias=True, padding_mode=padding_mode)

        self.ver_conv = nn.Conv2d(in_channels=cin, out_channels=cout, 
                                  kernel_size=(3, 1), stride=stride,
                                  padding=(1, 0), bias=True, padding_mode=padding_mode)

        self.hor_conv = nn.Conv2d(in_channels=cin, out_channels=cout, 
                                  kernel_size=(1, 3), stride=stride, 
                                  padding=(0, 1), bias=True, padding_mode=padding_mode)

        self.convCat = conv9x9(2,1)

        #self.conv9_1 = conv9x9(cin, cout)

        self.sigmoid = torch.nn.Sigmoid()
        self.depth = SELayer(cout)
        self.coSim = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.activation = nn.ReLU()
        self.norm = nn.InstanceNorm2d(cout)

    def forward(self, x):
        
        # Global Feature Extraction
        xsquare = self.square_conv(x)
        vertical_outputs = self.ver_conv(x)
        horizontal_outputs = self.hor_conv(x)

        
        # Sigmoid Gate
        
        '''avg_out = torch.mean(square_outputs, dim=1, keepdim=True)
        max_out, _ = torch.max(square_outputs, dim=1, keepdim=True)
        xsquare_cat = torch.cat([avg_out, max_out], dim=1)
        xsquare = self.sigmoid(self.convCat(x3_cat))'''

        avg_out = torch.mean(vertical_outputs, dim=1, keepdim=True)
        max_out, _ = torch.max(vertical_outputs, dim=1, keepdim=True)
        xvertical_cat = torch.cat([avg_out, max_out], dim=1)
        xvertical = self.sigmoid(self.convCat(xvertical_cat))

        avg_out = torch.mean(horizontal_outputs, dim=1, keepdim=True)
        max_out, _ = torch.max(horizontal_outputs, dim=1, keepdim=True)
        xhorizontal_cat = torch.cat([avg_out, max_out], dim=1)
        xhorizontal = self.sigmoid(self.convCat(xhorizontal_cat))


        x0 = xvertical + xhorizontal  
        x9 = self.depth(xsquare)
        x = x9 * x0
        return x   

class MultiKernelConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super(MultiKernelConvBlock, self).__init__()
        self.multikernelconv1 = MultiKernelConv2(cin, cout, stride=stride)
        self.activation = nn.LeakyReLU(0.2, inplace = True)
        self.norm = nn.InstanceNorm2d(cout)

    def forward(self, x):
        x1 = self.activation(self.multikernelconv1(x))
        #x2 = self.activation(self.multikernelconv2(x + x1))
        #x3 = self.activation(self.multikernelconv3(x + x1 + x2))
        x0 = x1 #+ x
        return x0