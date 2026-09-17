import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class PurifierCNN(nn.Module):
    """
    A lightweight U-Net style Convolutional Autoencoder for image purification.
    Expects 3-channel 224x224 input images.
    Uses skip connections to preserve high-frequency spatial details (edges, bones)
    while filtering out adversarial noise in the bottleneck.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(3, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.enc4 = self._conv_block(128, 256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(256, 512)
        
        # Decoder (with skip connections)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(512, 256)
        
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)
        
        # Final output layer
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # We don't use Sigmoid/Tanh because the input images are normalized (mean/std), 
        # so output can be negative or >1.
        return self.final_conv(d1)


# ---------------------------------------------------------
# Custom SSIM Implementation
# ---------------------------------------------------------
def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# ---------------------------------------------------------
# Combined Loss Module
# ---------------------------------------------------------
class CombinedLoss(nn.Module):
    """
    Combined Loss for Autoencoder: MSE + SSIM + Perceptual (VGG)
    """
    def __init__(self, mse_weight=1.0, ssim_weight=0.5, perceptual_weight=0.1, device='cuda'):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        self.device = device
        
        self.mse = nn.MSELoss()
        
        # Load pre-trained VGG16 for perceptual loss (feature extraction)
        vgg = models.vgg16(pretrained=True).features
        # We will extract features from the 3rd ReLU layer (block2_conv2)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:9]).to(device)
        self.feature_extractor.eval()
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, pred, target):
        # 1. MSE Loss (Pixel-level difference)
        loss_mse = self.mse(pred, target)
        
        # 2. SSIM Loss (Structural similarity)
        # SSIM returns [0, 1] where 1 is perfect match. Loss should be 1 - SSIM.
        loss_ssim = 1 - ssim(pred, target)
        
        # 3. Perceptual Loss (VGG Feature space difference)
        with torch.no_grad():
            target_features = self.feature_extractor(target)
        pred_features = self.feature_extractor(pred)
        loss_perceptual = self.mse(pred_features, target_features)
        
        # Combine
        total_loss = (self.mse_weight * loss_mse) + \
                     (self.ssim_weight * loss_ssim) + \
                     (self.perceptual_weight * loss_perceptual)
                     
        return total_loss, loss_mse, loss_ssim, loss_perceptual
