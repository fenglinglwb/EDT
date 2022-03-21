## On Efficient Transformer-Based Image Pre-training for Low-Level Vision 
---
Wenbo Li, Xin Lu, Shengju Qian, Jiangbo Lu, Xiangyu Zhang, Jiaya Jia
---
[\[Paper\]](https://arxiv.org/abs/2112.10175)

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

3. Download pretrained models from [One Drive](). Models are named by
   ```shell
   task_model_data[__pretrain_task_pretrain_data]
   ```
   where the optional part (in square brackets \[\]) indicates the pre-training setting. 
   - Task
      - Super-Resolution (SR) includes x2, x3, x4 scales.
      - Denoising (DN) includes Gaussian noise levels 15, 25 and 50, i.e., g15, g25, g50.
      - Deraining (DR) includees light and heavy streaks, i.e., ls and hs.
   - Type
      - Fine-tune: models are fine-tuned on target datasets with pre-training on ImageNet.
      - Pre-train: models are trained on ImageNet.
      - Scratch: models are trained on target datasets.
   - Model
      - EDT: T, S, B, L represent the tiny, small, base and large models.
      - EDTSF: SF means the denoising or deraining model without downsampling and upsampling in the encoder and decoder.
   - Datasets
      - Pre-train: ImageNet.
      - SR: Div2K, Flickr2K.
      - Denoising: Div2K, Flickr2K, BSD500 and WED, short for D4.
      - Deraining: RAIN100L, RAIN100H.
   - Note. We only provide pre-trained and fine-tuned models for deraining since RAIN100 dataests only contain hundreds of low-resolution images (insufficient to train the model). Especially, pre-trained models obtain better performance than fine-tuned ones. Thus, we report the evaluation results of pre-trained models in the paper. 
   <br />
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
       <td rowspan="22">Fine-tune</td>
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
      <td rowspan="5">Derain</td>
      <td rowspan="2">Fine-tune</td>
      <td>DRls_EDTB_RAIN100L__DRlshs_ImageNet200K</td>
    </tr>
    <tr>
      <td>DRhs_EDTB_RAIN100L__DRlshs_ImageNet200K</td>
    </tr>
    <tr>
      <td rowspan="3">Pre-train</td>
      <td>DRlshs_EDTB_ImageNet200K</td>
    </tr>
    <tr>
      <td>DRls_EDTB_ImageNet200K</td>
    </tr>
    <tr>
      <td>DRhs_EDTB_ImageNet200K</td>
    </tr>
   </tbody>
   </table> 


4. Quick test
