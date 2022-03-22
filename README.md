## On Efficient Transformer-Based Image Pre-training for Low-Level Vision 

#### Wenbo Li, Xin Lu, Shengju Qian, Jiangbo Lu, Xiangyu Zhang, Jiaya Jia
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
   task_model_data[__pretrain-task_pretrain-data]
   ```
   where the optional part (in square brackets \[\]) shows the pre-training setting. 
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
   - Note. We only provide pre-trained and fine-tuned models for deraining since RAIN100 dataests only contain hundreds of low-resolution images (insufficient to train transformers).
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
       <td rowspan="33">SR</td>
       <td rowspan="13">Fine-tune</td>
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
       <td>SRx2_EDTS_Div2kFlickr2K__SRx2_ImageNet200K</td>
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
       <td rowspan="12">Pre-train</td>
       <td>SRx2x3x4_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2x3DNg15_EDTB_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTT_ImageNet200K</td>
     </tr>
     <tr>
       <td>SRx2_EDTS_ImageNet200K</td>
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
       <td rowspan="8">Scratch</td>
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
      <td>DRhs_EDTB_RAIN100H__DRlshs_ImageNet200K</td>
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
    
    Put downloaded models into folder 'pretrained'. The model and config files are one-to-one with the same name. Please refer to the model list above.

    - SR and deraining.

    Read low-quality data directly from a specified folder as
    ```shell
    python test_sample.py --config config_path --model model_path --input input_folder [--output output_folder --gt gt_folder]
    ```
    where '--output' and '--gt' are optional. If assigned, the predictions will be stored and PSNR/SSIM results will be reported.

    For example,
    ```shell
    python test_sample.py --config configs/SRx2_EDTT_Div2kFlickr2K.py --model pretrained/SRx2_EDTT_Div2kFlickr2K.pth --input test_sets/SR/Set5/LR/x2 --gt test_sets/SR/Set5/HR/x2 
    ```

    - Denoising.

    The low-quality data are obtained by adding noise to the gt as
    ```shell
    python test_sample.py --config config_path --model model_path --gt gt_folder --noise_level XX [--output output_folder --sf]
    ```
    where 'sf' indicates whether there is upsampling and downsampling. If not assigned, EDT model will be built.

    For example,
    ```shell
    python test_sample.py --config configs/DNg15_EDTB_D4.py --model pretrained/DNg15_EDTB_D4.pth --gt test_sets/Denoise/McMaster --noise_level 15 
    ```

    - Note.

    The pre-training may contain multiple tasks. If you want to test multi-task models, please only build one branch and load corresponding weights during model building phase. We have provided an example for testing x2 SR based on model 'SRx2x3x4_EDTB_ImageNet200K' in the comment of 'test_sample.py'.

