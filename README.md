
# Audio to Face-Landmarks

![sadtalker](https://user-images.githubusercontent.com/4397546/222490039-b1f6156b-bf00-405b-9fda-0c9a9156f991.gif)


modify code CVPR2023 - [SadTalker](https://arxiv.org/abs/2211.12194) for generating landmarks from Audio



## 1. Installation.



1. Install [Anaconda](https://www.anaconda.com/), Python and `git`.

2. Creating the env and install the requirements.
  ```bash
  git clone https://github.com/OpenTalker/SadTalker.git

  cd SadTalker 

  conda create -n sadtalker python=3.8

  conda activate sadtalker
    
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

  conda install ffmpeg

  pip install -r requirements.txt
  
  pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1121/download.html
  
  python inference.py --driven_audio ./examples/driven_audio/RD_Radio31_000.wav --source_image ./examples/source_image/art_10.png --result_dir ./results --still --preprocess full --enhancer gfpgan


  ```  


## 2. Download Models

You can run the following script on Linux/macOS to automatically download all the models:

```bash
print('Download pre-trained models...')
rm -rf checkpoints
bash scripts/download_models.sh
```

We also provide an offline patch (`gfpgan/`), so no model will be downloaded when generating.


## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{zhang2022sadtalker,
  title={SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation},
  author={Zhang, Wenxuan and Cun, Xiaodong and Wang, Xuan and Zhang, Yong and Shen, Xi and Guo, Yu and Shan, Ying and Wang, Fei},
  journal={arXiv preprint arXiv:2211.12194},
  year={2022}
}
```
