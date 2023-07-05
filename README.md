# MedDeblur
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meddeblur-medical-image-deblurring-with/medical-image-deblurring-on-chexpert)](https://paperswithcode.com/sota/medical-image-deblurring-on-chexpert?p=meddeblur-medical-image-deblurring-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meddeblur-medical-image-deblurring-with/medical-image-deblurring-on-covid-19-ct-scan)](https://paperswithcode.com/sota/medical-image-deblurring-on-covid-19-ct-scan?p=meddeblur-medical-image-deblurring-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meddeblur-medical-image-deblurring-with/medical-image-deblurring-on-brain-mri)](https://paperswithcode.com/sota/medical-image-deblurring-on-brain-mri?p=meddeblur-medical-image-deblurring-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/meddeblur-medical-image-deblurring-with/medical-image-deblurring-on-human-protein)](https://paperswithcode.com/sota/medical-image-deblurring-on-human-protein?p=meddeblur-medical-image-deblurring-with)


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
 <img width=800 align="center" src = "https://github.com/sharif-apu/MedDeblur/blob/main/images/RD-SAM.png" alt="network"> </br>
</p>

**Figure:** Overview of proposed RD-SAM. It comprises a residual dense block, followed by a spatial-symmetric attention module. (a) Proposed RD-SAM. (b) Spatial-asymmetric attention module.


# Medical Image Deblurring Results </br>
**Qualitative Comparison** </br>
<p align="center">
<img width=800 align="center" src = "https://github.com/sharif-apu/MedDeblur/blob/main/images/vis_res.png" alt="Results"> </br>
</p>

**Figure:** </em> Performance of existing medical image deblurring methods in removing blind motion blur. The existing deblurring methods immensely failed in removing blur from medical images. (a) Blurry input. (b) Result obtained by TEMImageNet. (c) Result obtained by ZhaoNet. (d) Result obtained by Deep Deblur. (e) Result obtained by SRN Deblur [13]. (f) Proposed Method.

    
**Quantitative Comparison** </br>
<p align="center">
<img width=800 align="center"  src = "https://github.com/sharif-apu/MedDeblur/blob/main/images/results.png" alt="Results"> 
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

A few testing images are provided in a sub-directory under blurTest (i.e., blurTest/samples/)</br>
In such occasion, outputs will be available in modelOutput/sampless/. </br>

**To inference with custom setting execute the following command:**</br>
```python main.py -i -s path/to/inputImages -d path/to/outputImages``` </br>
Here, **-s** specifies the root directory of the source images
 (i.e., blurTest/), and **-d** specifies the destination root (i.e., modelOutput/).

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
