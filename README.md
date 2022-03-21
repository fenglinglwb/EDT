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
    <th>Model</th>
    <th>Pre-train</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">SR</td>
    <td>SRx2_EDTB_Div2kFlickr2K__SRx2x3x4_ImageNet200K</td>
    <td>Yes</td>
    <td>ha</td>
  </tr>
  <tr>
    <td>SRx3_EDTB_Div2kFlickr2K__SRx2x3x4_ImageNet200K</td>
    <td>Yes</td>
    <td>ha</td>
  </tr>
  <tr>
    <td>SRx4_EDTB_Div2kFlickr2K__SRx2x3x4_ImageNet200K</td>
    <td>Yees</td>
    <td>ha</td>
  </tr>
</tbody>
</table>

4. Quick test
    ```shell
    python3 test.py -- 
    ```
