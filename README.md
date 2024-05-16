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

- **Intuitive interface** for selecting objects using simple 2D points.
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

4. Add the path for **SegmentAnything** to your `init.py`:
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

## Quick Start

**SegmentAnything** allows artists to quickly create alpha mattes by adding just a few points to the desired object. 

To start, simply connect an input image and drag the pre-existing points onto the object you want to select. 

For better results, turn off the **Fast mode**, which helps the tool identify trickier objects.

### Improving Performance/Interactivity - ⚠️ Essential for Nuke 13 users

**SegmentAnything** requires a lot of processing power, even for the most modern GPUs. For a smoother experience, we recommend following this workflow, which involves pre-rendering the most memory-intensive step:

1. Create a **SegmentAnything** node and connect it to your input.
2. Click the **Export Encoded Matte** button and render the resulting **Write** node.
3. Load the rendered **encoded matte** and connect it to the **second input** of the **SegmentAnything** node.
4. Change the View back to **Final Output** and start selecting your objects.

Internally, **SegmentAnything** has two main steps: an encoder and a decoder. The encoder finds all the objects in a shot, while the decoder, based on the artist's input, recovers the desired object and converts it to an alpha matte. 

Since the **encoding** step doesn't need user input and is the most memory-intensive part of the process, pre-rendering it allows for nearly real-time feedback, as the decoder is much smaller in comparison.

##### Note for Nuke 13 users

While Nuke 14/15 users might be able to skip the pre-rendering step depending on their machine specs/GPUs, Nuke 13 makes this step **necessary** due to how slowly it processes large models. However, it still offers acceptable performance when using the pre-rendered **encoded matte**.

## Tips and Tricks

- **SegmentAnything** works best when selecting one object or class of objects at a time. For example, trying to select a person and a vehicle in the same shot using multiple points can confuse the model and give poor results.

- **SegmentAnything's** **encoding** step uses a lot of memory. To select multiple objects, pre-render the **encoding matte** and connect it to multiple **SegmentAnything** nodes.

- The **subtractive mode** works best on objects that are usually part of the same hierarchy. For example, if you want to select a person's t-shirt but the model selects the entire body, use subtractive points on the arms to indicate that you don't want the selection to go that far.

- For objects that aren't part of the same hierarchy - and the tool selects a second object by mistake due its proximity - it's usually better to create a separate node for the second object itself and mask it off using regular compositing operations.

- **SegmentAnything** wasn't designed for high-quality mattes, but the results can be improved with some filtering. To quickly enhance edge quality, check out the Guided Filter gizmo. Another option that uses a neural network for better edges is ViTMatte. Both are free and available in their respective repos.

## Options

![node_options](https://github.com/rafaelperez/Segment-Anything-for-Nuke/assets/1684365/6ae0e629-6cc6-46a0-8453-78a0a0933f35)

- **View**: Determines the output of the node.
  - **Final Result:** Generates the matte as an alpha channel.
  - **Encoded Matte:** Outputs a pre-processed matte that can be used as an input to speed up interactivity. The same encoded matte can be connected to multiple SegmentAnything nodes for selecting different objects in the same scene.

- **Fast Mode:** Speeds up processing and reduces memory usage at the cost of a lower quality model.

- **Bypass sRGB conversion:** SegmentAnything expects sRGB images for optimal results. Enable this option to use your own color space conversion.

- **Add Point:** Adds a selection point (up to 8 points supported).

- **Remove Point:** Removes the last added selection point.

- **Reset All:** Clears all selection points.

- **Subtractive mode:** Excludes an area from the matte. Use this only if the initial selection spills over into an unwanted region. For better results, create a separate selection for the unwanted object and mask it from the original.

- **Overlay Color:** Adjusts the mask overlay color.

- **Disable Overlay:** Turns off the matte overlay.

- **Export Encoded Matte:** Creates a `Write` node and sets the output to `Encoded Matte`. The rendered matte can be connected to this node's second input to significantly improve interactivity.

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
