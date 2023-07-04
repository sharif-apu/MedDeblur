# MedDeblur

This is the official implementation of a state-of-the-art medical image deblurring method titled as **"MedDeblur: Medical Image Deblurring with Residual Dense Spatial-Asymmetric Attention"**. **[[Click Here]([https://www.mdpi.com/2227-7390/8/12/2192/pdf](https://www.mdpi.com/2227-7390/11/1/115/pdf?version=1672128173))]** to download the full paper (in PDF).  </br>

**Please consider citing this paper as follows:**
```
@article{sharif2022meddeblur,
  title={MedDeblur: Medical Image Deblurring with Residual Dense Spatial-Asymmetric Attention},
  author={Sharif, SMA and Naqvi, Rizwan Ali and Mehmood, Zahid and Hussain, Jamil and Ali, Ahsan and Lee, Seung-Won},
  journal={Mathematics},
  volume={11},
  number={1},
  pages={115},
  year={2022},
  publisher={MDPI}
}

```


# Network Architecture
# Multi-scale Network
<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/MedDeblur/blob/main/images/overview.png" alt="Overview"> </br>
</p>

**Figure:** Overview of the proposed network for learning medical image deblurring. The proposed method comprises a novel RD-SAM block in a scale recurrent network for learning salient features to accelerate deblurring performance. 

# Proposed RD-SAM

<p align="center">
 <img width=800 align="center" src = "[https://user-images.githubusercontent.com/15001857/101642681-79841180-3a5d-11eb-9dcf-ae9db1e757f9.png](https://github.com/sharif-apu/MedDeblur/blob/main/images/RD-SAM.png)" alt="network"> </br>
</p>

**Figure:** Overview of proposed RD-SAM. It comprises a residual dense block, followed by a spatial-symmetric attention module. (a) Proposed RD-SAM. (b) Spatial-asymmetric attention module.


# Medical Image Deblurring Results </br>
**Qualitative Comparison** </br>
<p align="center">
<img width=800 align="center" src = "https://user-images.githubusercontent.com/15001857/101643040-e697a700-3a5d-11eb-8099-e054ae9c7759.png" alt="Results"> </br>
</p>

**Figure:** </em> Performance of existing medical image deblurring methods in removing blind motion blur. The existing deblurring methods immensely failed in removing blur from medical images. (a) Blurry input. (b) Result obtained by TEMImageNet. (c) Result obtained by ZhaoNet. (d) Result obtained by Deep Deblur. (e) Result obtained by SRN Deblur [13]. (f) Proposed Method.

    
**Quantitative Comparison** </br>
<p align="center">
<img width=800 align="center"  src = "https://user-images.githubusercontent.com/15001857/101272591-c9da4580-37b7-11eb-8db8-37d7c53ed36c.png" alt="Results"> 
</p>

**Table:** Objective comparison between deep deblurring methods for MID. We evaluated the performance of each comparing method by utilizing the evaluation metrics. Moreover, we calculated individual scores (i.e., PSNR, SSIM, and deltaE) for all testing images. We compute the mean performance of each comparing method for a specific dataset to observe their performance on that respective modality. Later, we summarized the performance of each comparing method by calculating the mean PSNR, SSIM, and deltaE scores obtained on the individual modality.
 </br>



# Prerequisites
```
Python 3.8
CUDA 10.1 + CuDNN
pip
Virtual environment (optional)
```

# Installation
**Please consider using a virtual environment to continue the installation process.**
```
git clone https://github.com/sharif-apu/MID-DRAN.git
cd MID-DRAN
pip install -r requirement.txt
```

# Testing
**MeDeblur can be inferenced with pretrained weights and default setting as follows:** </br>
```python main.py -i``` </br>

A few testing images are provided in a sub-directory under testingImages (i.e., testingImages/sampleImages/)</br>
In such occasion, denoised image(s) will be available in modelOutput/sampleImages/. </br>

**To inference with custom setting execute the following command:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages -ns=sigma(s)``` </br>
Here,**-ns** specifies the standard deviation of a Gaussian distribution (i.e., -ns=15, 25, 50),**-s** specifies the root directory of the source images
 (i.e., testingImages/), and **-d** specifies the destination root (i.e., modelOutput/).

# Training
**To train with your own dataset execute:**</br>
```python main.py -ts -e X -b Y```
To specify your trining images path, go to mainModule/config.json and update "trainingImagePath" entity. </br>You can specify the number of epoch with **-e** flag (i.e., -e 5) and number of images per batch with **-b** flag (i.e., -b 24).</br>

**For transfer learning execute:**</br>
```python main.py -tr -e -b ```

# Others
**Check model configuration:**</br>
```python main.py -ms``` </br>
**Create new configuration file:**</br>
```python main.py -c```</br>
**Update configuration file:**</br>
```python main.py -u```</br>
**Overfitting testing** </br>
```python main.py -to ```</br>

# Contact
For any further queries, feel free to contact us through the following emails: apuism@gmail.com, rizwanali@sejong.ac.kr
