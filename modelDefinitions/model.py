 
import torch
import torch.nn as nn
from modelDefinitions.layers import *
from modelDefinitions.conv_lstm import CLSTM_cell

def conv5x5_relu(in_channels , out_channels , stride ):
    return conv(in_channels , out_channels , 5 , stride , 2 , activation_fn = partial( nn.ReLU , inplace = True )  )

def deconv5x5_relu( in_channels , out_channels , stride , output_padding ):
    return deconv(in_channels , out_channels , 5 , stride , 2 , output_padding = output_padding ,  activation_fn = partial( nn.ReLU , inplace = True ) )

def resblock(in_channels ):
    """Resblock without BN and the last activation
    """
    return BasicBlock(in_channels , out_channels = in_channels , kernel_size = 5 , stride = 1 , use_batchnorm = False , activation_fn = partial(nn.ReLU , inplace = True) , last_activation_fn = None )

class EBlock(nn.Module):
    def __init__( self , in_channels , out_channels , stride, number_block = 1):
        super(type(self),self).__init__()
        self.conv = conv5x5_relu( in_channels , out_channels , stride )
        resblock_list = []
        for i in range( number_block ):
            resblock_list.append( RDB(out_channels, 3, 8) )
        self.resblock_stack = nn.Sequential( *resblock_list )
        self.sam = MultiKernelConvBlock(out_channels, out_channels)

    def forward( self , x):
        x = self.conv(x)
        x = self.resblock_stack(x)
        x = self.sam(x)
        return x
        
class DBlock(nn.Module):
    def __init__( self , in_channels , out_channels , stride , output_padding, number_block = 1):
        super(type(self),self).__init__()
        resblock_list = []
        for i in range( number_block ):
            resblock_list.append( RDB(in_channels, 3, 8) )
        self.resblock_stack = nn.Sequential( *resblock_list )
        self.sam = MultiKernelConvBlock(in_channels, in_channels)
        self.deconv = deconv5x5_relu( in_channels , out_channels , stride , output_padding )
        
    def forward( self , x ):
        x = self.resblock_stack( x )
        x = self.sam(x)
        x = self.deconv( x )
        return x

class OutBlock(nn.Module):
    def __init__( self, in_channels, number_block = 1 ):
        super(type(self),self).__init__()
        resblock_list = []
        for i in range( number_block ):
            resblock_list.append( RDB(in_channels, 3, 8)  )
        self.resblock_stack = nn.Sequential( *resblock_list )
        self.sam = MultiKernelConvBlock(in_channels, in_channels)
        self.conv = conv( in_channels , 3 , 5 , 1 , 2  , activation_fn = None ) 
    def forward( self , x ):
        x = self.resblock_stack( x )
        x = self.sam(x)
        x = self.conv( x )
        return x

class SRNDeblurNet(nn.Module):
    """SRN-DeblurNet 
    examples:
        net = SRNDeblurNet()
        y = net( x1 , x2 , x3）#x3 is the coarsest image while x1 is the finest image
    """

    def __init__( self  , upsample_fn = partial( torch.nn.functional.interpolate , mode = 'bilinear',align_corners=False ) , xavier_init_all = True, number_block = 3 ):
        super(type(self),self).__init__()
        self.upsample_fn = upsample_fn
        self.inblock = EBlock( 3 + 3 , 32 , 1, number_block  )
        self.eblock1 = EBlock( 32 , 64 , 2, number_block )
        self.eblock2 = EBlock( 64 , 128 , 2, number_block )
        self.convlstm = CLSTM_cell( 128 , 128 , 5 )
        self.dblock1 = DBlock( 128 , 64 , 2  , 1, number_block)
        self.dblock2 = DBlock( 64 , 32 , 2  , 1, number_block)
        self.outblock = OutBlock( 32, number_block )

        self.input_padding  = None
        if xavier_init_all:
            for name,m in self.named_modules():
                if isinstance( m , nn.Conv2d ) or isinstance(m , nn.ConvTranspose2d ):
                    torch.nn.init.xavier_normal_(m.weight)
                    #print(name)

    def forward_step( self , x , hidden_state ):
        e32 = self.inblock( x )
        e64 = self.eblock1( e32 )
        e128 = self.eblock2( e64 )
        h,c = self.convlstm( e128 , hidden_state )
        d64 = self.dblock1( h )
        d32 = self.dblock2( d64 + e64 )
        d3 = self.outblock( d32 + e32 )
        return d3 , h,c
        
    def forward( self , b1, b2, b3  ):
        #b1 , b2 , b3 = torch.rand(2, 3, 64, 64).cuda(), torch.rand(2, 3, 32, 32).cuda(), torch.rand(2, 3, 16, 16).cuda()
        #print(b1.shape, b2.shape, b3.shape) 
        if self.input_padding is None or self.input_padding.shape != b3.shape:
            self.input_padding = torch.zeros_like( b3  )
        h,c = self.convlstm.init_hidden(b3.shape[0],(b3.shape[-2]//4,b3.shape[-1]//4))

        i3 , h , c = self.forward_step( torch.cat( [b3 , self.input_padding ] , 1 ) , (h,c) )

        c = self.upsample_fn( c , scale_factor = 2 )
        h = self.upsample_fn( h , scale_factor = 2 )
        i2 , h , c = self.forward_step( torch.cat( [b2 , self.upsample_fn( i3 , scale_factor = 2 ) ] , 1) , (h,c) )

        c = self.upsample_fn( c , scale_factor = 2  )
        h = self.upsample_fn( h , scale_factor = 2  )
        i1 , h,c = self.forward_step( torch.cat( [b1 , self.upsample_fn( i2 , scale_factor = 2 ) ] , 1) , (h,c) )

        #y2 = self.upsample_fn( y1 , (128,128) ) 
        #y3 = self.upsample_fn( y2 , (64,64) )

        #return y1 , y2 , y3
        return torch.tanh(i1), i2, i3 

from torchsummary import summary
from pthflops import count_ops
if __name__ == "__main__":
    net = SRNDeblurNet().cuda()
    summary(net, input_size = (3, 64, 64))
    device = 'cuda:0'
    model = SRNDeblurNet().to(device)
    inp = torch.rand(1,3,128,128).to(device)

    # Count the number of FLOPs
    count_ops(model, inp)