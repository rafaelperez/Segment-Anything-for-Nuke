# Segment Anything (SAM) for Nuke

## Introduction

This project brings Meta's powerful **Segment Anything Model (SAM)** to **The Foundry's Nuke**. **Segment Anything** is a state-of-the-art neural network for creating precise masks around objects in single images, capable of handling both familiar and unfamiliar subjects without additional training.

This project offers a native integration within Nuke, requiring no external dependencies or complex installation. The neural network is wrapped into an intuitive **Gizmo**, controllable via Nuke's standard Tracker for a seamless experience.

With this implementation, you gain access to cutting-edge object segmentation capabilities directly inside your Nuke workflow, leveraging **Segment Anything** to isolate and extract objects in time efficinet manner.  streamlining your compositing tasks.

<div align="center">

[![author](https://img.shields.io/badge/by:_Rafael_Silva-red?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rafael-silva-ba166513/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>


## Features

- **Intuitive interface** for selecting objects using Nuke's familiar Tracker node.
- **Fast mode**, allowing users to balance precision and GPU memory usage.
- **Preprocessing stage** with an encoded matte for reusing and speeding up multiple object selections.
- **Efficient memory usage** - the high-quality model fits on most 8GB graphics cards, while the low-quality model is compatible with 4GB cards.
- **Nuke 13 compatibility**. Note: **Preprocessing is recommended** for an optimal experience.

## Compatibility

**Nuke 13.2+**, tested on **Linux** and **Windows**.

## Installation

1. Download and unzip the latest release from [here](https://github.com/rafaelperez/Segment-Anything-for-Nuke/releases).
2. Copy the extracted `Cattery` folder to `.nuke` or your plugins path.
3. In the toolbar, choose **Cattery > Update** or simply **restart** Nuke.

**Segment Anything** will then be accessible under the toolbar at **Cattery > Segmentation > SegmentAnything**.

### ⚠️ Extra Steps for Nuke 13

4. Add the path for **RIFE** to your `init.py`:
``` py
import nuke
nuke.pluginAddPath('./Cattery/SegmentAnything')
```

5. Add an menu item to the toolbar in your `menu.py`:

``` py
import nuke
toolbar = nuke.menu("Nodes")
toolbar.addCommand('Cattery/Segmentation/SegmentAnything', 'nuke.createNode("SAM")', icon="SAM.png")
```

## Compiling the Model

To retrain or modify the model for use with **Nuke's CatFileCreator**, you'll need to convert it into the PyTorch format `.pt`. Below are the primary methods to achieve this:

### Cloud-Based Compilation (Recommended for Nuke 14+)

**Google Colaboratory** offers a free, cloud-based development environment ideal for experimentation or quick modifications. It's important to note that Colaboratory uses **Python 3.10**, which is incompatible with the **PyTorch version (1.6)** required by **Nuke 13**.

For those targetting **Nuke 14** or **15**, [Google Colaboratory](https://colab.research.google.com) is a convenient choice.

### Local Compilation (Required for Nuke 13+)

Compiling the model locally gives you full control over the versions of **Python**, **PyTorch**, and **CUDA** you use. Setting up older versions, however, can be challenging.

For **Nuke 13**, which requires **PyTorch 1.6**, using **Docker** is highly recommended. This recommendation stems from the lack of official PyTorch package support for **CUDA 11**.

Fortunately, Nvidia offers Docker images tailored for various GPUs. The Docker image version **20.07** is specifically suited for **PyTorch 1.6.0 + CUDA 11** requirements.

Access to these images requires registration on [Nvidia's NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

Once Docker is installed on your system, execute the following command to initiate a terminal within the required environment. You can then clone the repository and run `python sam_nuke.py` to compile the model.

```sh
docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:20.07-py3
git clone https://github.com/rafaelperez/Segment-Anything-for-Nuke.git
cd Segment-Anything-for-Nuke
python sam_nuke.py
```
For projects targeting **Nuke 14+**, which requires **PyTorch 1.12**, you can use the following Docker image, version **22.05**:

`docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:22.05-py3`

For more information on selecting the appropriate Python, PyTorch, and CUDA combination, refer to [Nvidia's Framework Containers Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2020).

## License and Acknowledgments

**SegmentAnything.cat** is licensed under the MIT License, and is derived from https://github.com/facebookresearch/segment-anything.

While the MIT License permits commercial use of **ViTMatte**, the dataset used for its training may be under a non-commercial license.

This license **does not cover** the underlying pre-trained model, associated training data, and dependencies, which may be subject to further usage restrictions.

Consult https://github.com/facebookresearch/segment-anything for more information on associated licensing terms.

**Users are solely responsible for ensuring that the underlying model, training data, and dependencies align with their intended usage of RIFE.cat.**

## Citation

If you use SAM or SA-1B in your research, please use the following BibTeX entry.

```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
