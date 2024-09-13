# ActorsNeRF: Animatable Few-shot Human Rendering with Generalizable NeRFs (ICCV 2023)

[**ActorsNeRF: Animatable Few-shot Human Rendering with Generalizable NeRFs (ICCV 2023)**](https://openaccess.thecvf.com/content/ICCV2023/papers/Mu_ActorsNeRF_Animatable_Few-shot_Human_Rendering_with_Generalizable_NeRFs_ICCV_2023_paper.pdf)
<br>
[*JitengMu*](https://jitengmu.github.io/), [*Shen Sang*](https://ssangx.github.io/), [*Nuno Vasconcelos*](http://www.svcl.ucsd.edu/~nuno/), [*Xiaolong Wang*](https://xiaolonw.github.io/)
<br>
ICCV 2023

The project page with more details is at [https://jitengmu.github.io/ActorsNeRF/](https://jitengmu.github.io/ActorsNeRF/).

<div align="center">
<img src="figs/demo.gif" width="75%">
</div>


## Citation

If you find our code or method helpful, please use the following BibTex entry.
```
@article{mu2023actorsnerf,
  author    = {Jiteng Mu and
               Shen Sang and
               Nuno Vasconcelos and
               Xiaolong Wang},
  title     = {ActorsNeRF: Animatable Few-shot Human Rendering with Generalizable NeRFs},
  booktitle = {ICCV},
  pages = {18345--18355},
  year      = {2023},
}
```

This is an official implementation. The codebase is implemented using [PyTorch](https://pytorch.org/) and tested on [Ubuntu](https://ubuntu.com/) 20.04.4 LTS.

## Prerequisite

### Environment

Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [Anaconda](https://www.anaconda.com/).

Create and activate a virtual environment.

    conda create --name actorsnerf python=3.7
    conda activate actorsnerf

Install the required packages.

    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install -r requirements.txt


### Dataset

We follow HumanNeRF to preprocess the ZJU-Dataset and AIST++ Dataset. Please download the dataset from [ZJU-MoCap Dataset](https://github.com/zju3dv/neuralbody) and [AIST++ Dataset](https://google.github.io/aistplusplus_dataset/factsfigures.html) accordingly. We follow [Nueral Body](https://github.com/zju3dv/neuralbody) and [HumanNeRF](https://github.com/chungyiweng/humannerf) to preprocess datasets and we provide our preprocessing scripts under `tools/`. Please contact the authors Jiteng Mu (jmu@ucsd.edu) for the processed dataset. The dataset is put under `./datasets` and organized as following,

```
datasets/
    zju_mocap/
        lbs/
        new_vertices/
        313/
            0/
                cameras.pkl
                canonical_joints.pkl
                images
                masks
                mesh_infos.pkl
            1/
                cameras.pkl
                canonical_joints.pkl
                images
                masks
                mesh_infos.pkl
            ...
        315/
        ...
    AIST_mocap
        lbs/
        new_vertices/
        d01/
            0/
                cameras.pkl
                canonical_joints.pkl
                images
                masks
                mesh_infos.pkl
            1/
                cameras.pkl
                canonical_joints.pkl
                images
                masks
                mesh_infos.pkl
            ...
        d02/
        ...
```


## Training and Inference

### `1. Category-level Training`

For ZJU-MoCap Dataset,

    python3 train.py --cfg configs/actorsnerf/zju_mocap/zju_category_level-pretrain.yaml

For AIST++ Dataset,

    python3 train.py --cfg configs/actorsnerf/AIST_mocap/AIST_category_level-pretrain.yaml

### `2. Few-shot Optimization`

For ZJU-Mocap Dataset, 10-shot setting on actor 387,

    python3 train.py --cfg configs/actorsnerf/zju_mocap/387/zju_387-10_shot-skip30.yaml

For AIST++ Dataset, 30-shot setting on actor d20,

    python3 train.py --cfg configs/actorsnerf/AIST_mocap/d20/AIST_d20-10_shot-skip30.yaml

You may find config files for other settings under `configs/actorsnerf`

### `3. Evaluation and Rendering on novel views and poses`

Run free-viewpoint rendering on novel views and novel poses. The following script will run evaluation on actor d20 under the 10-shot setting. Results are saved to `experiments/actorsnerf/AIST_mocap/d20/AIST_d20-10_shot-skip30/` by default.

    python eval.py --type eval_novel_view --cfg configs/actorsnerf/AIST_mocap/d20/AIST_d20-10_shot-skip30.yaml

Please download the [checkpoints](https://huggingface.co/datasets/JitengMu/ICCV2023_ActorsNeRF_release), where we also provide the produced images.

## Acknowledgement

The implementation took reference from [HumanNeRF](https://github.com/chungyiweng/humannerf), [Neural Body](https://github.com/zju3dv/neuralbody), [LPIPS](https://github.com/richzhang/PerceptualSimilarity). We thank the authors for their generosity to release code.
