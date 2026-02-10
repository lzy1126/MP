# MP# MP
## Related Paper

This repository contains the official PyTorch implementation for the paper:

**Enhanced Font Generation through Multi-Scale Attention and Projection-Based Character Loss**

Zhongyu Li, Bodong Li  


Prerequisites (Recommended)
- Linux
- Python 3.9
- Pytorch 1.13.1
- CUDA 11.7

**Step 0**: Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html),Download FFT file https://www.foundertype.com/.

**Step 1**: Create a conda environment and activate it.
```bash
conda create -n font python=3.9 -y
conda activate font
```

**Step 2**: Install related version Pytorch following [here](https://pytorch.org/get-started/previous-versions/).
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

**Step 3**: Install the required packages.
```bash
pip install -r requirements.txt
```

### Training
### Data Construction
The training data files tree should be (The data examples are shown in directory `data_examples/train/`):
```
├──data_examples
│   └── train
│       ├── ContentImage
│       │   ├── char0.png
│       │   ├── char1.png
│       │   ├── char2.png
│       │   └── ...
│       └── TargetImage.png
│           ├── style0
│           │     ├──style0+char0.png
│           │     ├──style0+char1.png
│           │     └── ...
│           ├── style1
│           │     ├──style1+char0.png
│           │     ├──style1+char1.png
│           │     └── ...
│           ├── style2
│           │     ├──style2+char0.png
│           │     ├──style2+char1.png
│           │     └── ...
│           └── ...
```
### Training 
```bash
sh train.sh
```
- `data_root`: The data root, as `./data_examples`
- `output_dir`: The training output logs and checkpoints saving directory.
- `resolution`: The resolution of the UNet in our diffusion model.
- `style_image_size`: The resolution of the style image, can be different with `resolution`.
- `content_image_size`: The resolution of the content image, should be the same as the `resolution`.
- `channel_attn`: Whether to use the channel attention in the MAF block.
- `train_batch_size`: The batch size in the training.
- `max_train_steps`: The maximum of the training steps.
- `learning_rate`: The learning rate when training.
- `ckpt_interval`: The checkpoint saving interval when training.
- `drop_prob`: The classifier-free guidance training probability.


##  Sampling
### Step 1 => Prepare the checkpoint   
Put your re-training checkpoint folder `ckpt` to the root directory, including the files `unet.pth`, `content_encoder.pth`, and `style_encoder.pth`.

### Step 2 => Run the script  
```bash
sh script/sample_content_image.sh
```
- `ckpt_dir`: The model checkpoints saving directory.  
- `content_image_path`: The content/source image path.
- `style_image_path`: The style/reference image path.
- `save_image`: set `True` if saving as images.
- `save_image_dir`: The image saving directory, the saving files including an `out_single.png` and an `out_with_cs.png`.
- `device`: The sampling device, recommended GPU acceleration.
- `guidance_scale`: The classifier-free sampling guidance scale.
- `num_inference_steps`: The inference step by DPM-Solver++.
