# Vision-Language FSOD
[![arXiv](https://img.shields.io/badge/arXiv-2312.14494-b31b1b.svg)](https://arxiv.org/abs/2312.14494)
[![models](https://img.shields.io/badge/🤗HuggingFace-Model-yellow)](https://huggingface.co/empMEMORY/vision-language-fsod/tree/main)
[![challenge](https://img.shields.io/badge/EvalAI-FSOD_Challenge-green)](https://eval.ai/web/challenges/challenge-page/2270/overview)

![teaser.png](assets/teaser.png)

## :star: [FSOD Challenge](https://eval.ai/web/challenges/challenge-page/2270/overview)
This repository is part of the FSOD Challenge at CVPR 2024.

## Abstract
This repository features a new Few-shot object detection (FSOD) benchmark protocol, **Foundational FSOD**, that evaluates detectors pre-trained on any external datasets and fine-tuned on K-shots per target class.

## Installation
See [installation instructions](docs/INSTALL.md).

## Data
See [datasets/README.md](datasets/README.md)

## Models
Create `models/` in the root directory and download pre-trained model [here](https://huggingface.co/empMEMORY/vision-language-fsod/tree/main/pretrained_models/)

## Training
```python
python train_net.py --num-gpus 1 --config-file <config_path>  --pred_all_class  OUTPUT_DIR_PREFIX <root_output_dir>
```

## Inference
```python 
python train_net.py --num-gpus 8 --config-file <config_path>  --pred_all_class --eval-only  MODEL.WEIGHTS <model_path> OUTPUT_DIR_PREFIX <root_output_dir>
```

## TODO
- [x] Code cleanup 
- [x] Release FSOD training files 
- [ ] FIOD support and config
- [ ] FSOD Data split creation : nuImages along with new split
- [ ] Release trained model 

- ------------
- [ ] LVIS support in data and training models

## Acknowledgment
We thank the authors of the following repositories for their open-source implementations which were used in building the current codebase:
1. [Detic: Detecting Twenty-thousand Classes using Image-level Supervision](https://github.com/facebookresearch/Detic)
2. [Detectron2](https://github.com/facebookresearch/detectron2)

## Citation
If you find our paper and code repository useful, please cite us:
```bib
@article{madan2023revisiting,
  title={Revisiting Few-Shot Object Detection with Vision-Language Models},
  author={Madan, Anish and Peri, Neehar and Kong, Shu and Ramanan, Deva},
  journal={arXiv preprint arXiv:2312.14494},
  year={2023}
}
```