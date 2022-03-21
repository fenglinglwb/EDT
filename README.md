## On Efficient Transformer-Based Image Pre-training for Low-Level Vision 
Wenbo Li, Xin Lu, Shengju Qian, Jiangbo Lu, Xiangyu Zhang, Jiaya Jia
---
\[Paper\](https://arxiv.org/abs/2112.10175)

We have made the testing code and well-trained models for SR, denoising and deraining available now. The training code will be released soon.


---
### Usage

1. Clone the repository
    ```shell
    git clone https://github.com/fenglinglwb/EDT.git 
    ```
2. Install the dependencies
    - Python >= 3.7
    - PyTorch >= 1.4
    - Other packages
    ```shell
    pip install -r requirements.txt
    ```

3. Download pretrained models from [One Drive](). Models are named by task\_model\_data\[\_\_pretrain\_task\_pretrain\_data\] where the optional denotation indicates the pre-training setting. 
   <table>
   <thead>
     <tr>
       <th>Task</th>
       <th>Type</th>
       <th>Model</th>
     </tr>
   </thead>
   <tbody>
     <tr>
       <td rowspan="53">SR</td>
       <td rowspan="22">Fine-tune<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br></td>
       <td>SRx2_EDTT_Div2kFlickr2K__SRx2x3x4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTT_Div2kFlickr2K__SRx2x3x4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx4_EDTT_Div2kFlickr2K__SRx2x3x4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_Div2kFlickr2K__SRx2x3x4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTB_Div2kFlickr2K__SRx2x3x4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx4_EDTB_Div2kFlickr2K__SRx2x3x4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTT_Div2kFlickr2K__SRx2_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTT_Div2kFlickr2K__SRx3_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx4_EDTT_Div2kFlickr2K__SRx4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTS_Div2kFlickr2K__SRx2_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTS_Div2kFlickr2K__SRx3_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx4_EDTS_Div2kFlickr2K__SRx4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_Div2kFlickr2K__SRx2_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTB_Div2kFlickr2K__SRx3_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx4_EDTB_Div2kFlickr2K__SRx4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTL_Div2kFlickr2K__SRx2_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTL_Div2kFlickr2K__SRx3_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx4_EDTL_Div2kFlickr2K__SRx4_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_Div2kFlickr2K__SRx2_ImageNet50K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_Div2kFlickr2K__SRx2_ImageNet100K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_Div2kFlickr2K__SRx2_ImageNet400K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_Div2kFlickr2K__SRx2_ImageNetFull</td>
     </tr>
     <tr>
       <td rowspan="19">Pre-train</td>
       <td>SRx2x3x4_EDTT_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2x3x4_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2x3DNg15_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTT_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTT_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx4_EDTT_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTS_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTS_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx4_EDTS_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx4_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTL_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTL_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx3_EDTL_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_ImageNet50K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_ImageNet100K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_ImageNet400K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_ImageNetFull</td>
     </tr>
     <tr>
       <td rowspan="12">Scratch</td>
       <td>SRx2_EDTT_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx3_EDTT_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx4_EDTT_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx2_EDTS_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx3_EDTS_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx4_EDTS_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx2_EDTB_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx3_EDTB_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx4_EDTB_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx2_EDTL_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx3_EDTL_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td>SRx4_EDTL_Div2kFlickr2K</td>
     </tr>
     <tr>
       <td rowspan="16">Denoise</td>
       <td rowspan="6">Fine-tune</td>
       <td>DNg15_EDTB_D4__DNg15g25g50_ImageNet200K</td>
     </tr>
     <tr>
       <td>DNg25_EDTB_D4__DNg15g25g50_ImageNet200K</td>
     </tr>
     <tr>
       <td>DNg50_EDTB_D4__DNg15g25g50_ImageNet200K</td>
     </tr>
     <tr>
       <td>DNg15_EDTB_D4__DNg15_ImageNet200K</td>
     </tr>
     <tr>
       <td>DNg25_EDTB_D4__DNg25_ImageNet200K</td>
     </tr>
     <tr>
       <td>DNg50_EDTB_D4__DNg50_ImageNet200K</td>
     </tr>
     <tr>
       <td rowspan="4">Pre-train</td>
       <td>DNg15g25g50_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>DNg15_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>DNg25_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>DNg50_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td rowspan="6">Scratch</td>
       <td>DNg15_EDTB_D4</td>
     </tr>
     <tr>
       <td>DNg25_EDTB_D4</td>
     </tr>
     <tr>
       <td>DNg50_EDTB_D4</td>
     </tr>
     <tr>
       <td>DNg15_EDTBSF_D4</td>
     </tr>
     <tr>
       <td>DNg25_EDTBSF_D4</td>
     </tr>
     <tr>
       <td>DNg50_EDTBSF_D4</td>
     </tr>
     <tr>
       <td rowspan="3">Derain</td>
       <td rowspan="2">Fine-tune</td>
       <td>DRls_EDTB_RAIN100L__DRlshs_ImageNet200K</td>
     </tr>
     <tr>
       <td>DRhs_EDTB_RAIN100L__DRlshs_ImageNet200K</td>
     </tr>
     <tr>
       <td>Pre-train</td>
       <td>DRlshs_EDTB_ImageNet200K</td>
     </tr>
   </tbody>
   </table> 


4. Quick test
