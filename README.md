<img src="PanopticBEV/images/cvlab.png" align="right" width=8%>

# [SeaBird: Segmentation in Bird's View with Dice Loss Improves Monocular 3D Detection of Large Objects](https://arxiv.org/pdf/2403.20318.pdf)

### [KITTI-360 Demo](https://www.youtube.com/watch?v=SmuRbMbsnZA) | [nuScenes Demo] | [Project](http://cvlab.cse.msu.edu/project-seabird.html) | [Talk] | [Slides] | [Poster]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seabird-segmentation-in-bird-s-view-with-dice/3d-object-detection-from-monocular-images-on-7)](https://paperswithcode.com/sota/3d-object-detection-from-monocular-images-on-7?p=seabird-segmentation-in-bird-s-view-with-dice) 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/seabird-segmentation-in-bird-s-view-with-dice/3d-object-detection-on-nuscenes-camera-only)](https://paperswithcode.com/sota/3d-object-detection-on-nuscenes-camera-only?p=seabird-segmentation-in-bird-s-view-with-dice)

[![arXiv](http://img.shields.io/badge/arXiv-2403.20318-B31B1B.svg)](https://arxiv.org/abs/2403.20318)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Visitors](https://api.visitorbadge.io/api/visitors?path=abhi1kumar%2FSeaBird&labelColor=%23FFFFFF&countColor=%23721e82&style=flat)](https://visitorbadge.io/status?path=abhi1kumar%2FSeaBird)
[![GitHub Stars](https://img.shields.io/github/stars/abhi1kumar/SeaBird?style=social)](https://github.com/abhi1kumar/SeaBird)

[Abhinav Kumar](https://sites.google.com/view/abhinavkumar)<sup>1</sup>, 
[Yuliang Guo](https://yuliangguo.github.io)<sup>2</sup>, 
[Xinyu Huang](https://scholar.google.com/citations?user=cL4bNBwAAAAJ&hl=en)<sup>2</sup>, 
[Liu Ren](https://www.liu-ren.com)<sup>2</sup>, 
[Xiaoming Liu](http://www.cse.msu.edu/~liuxm/index2.html)<sup>1</sup> <br>
<sup>1</sup>Michigan State University, <sup>2</sup>Bosch Research North America, Bosch Center for AI

in [CVPR 2024](https://cvpr.thecvf.com/Conferences/2024/)

<img src="PanopticBEV/images/Seabird_pipeline.png" width="1024">

> Monocular 3D detectors achieve remarkable performance on cars and smaller objects. However, their performance drops on larger objects, leading to fatal accidents. Some attribute the failures to training data scarcity or the receptive field requirements of large objects. In this paper, we highlight this understudied problem of generalization to large objects. We find that modern frontal detectors struggle to generalize to large objects even on nearly balanced datasets. We argue that the cause of failure is the sensitivity of depth regression losses to noise of larger objects. To bridge this gap, we comprehensively investigate regression and dice losses, examining their robustness under varying error levels and object sizes. We mathematically prove that the dice loss leads to superior noise-robustness and model convergence for large objects compared to regression losses for a simplified case. Leveraging our theoretical insights, we propose SeaBird (Segmentation in Bird's View) as the first step towards generalizing to large objects. SeaBird effectively integrates BEV segmentation on foreground objects for 3D detection, with the segmentation head trained with the dice loss. SeaBird achieves SoTA results on the KITTI-360 leaderboard and improves existing detectors on the nuScenes leaderboard, particularly for large objects.

<img src="PanopticBEV/images/seabird_kitti360_demo.gif" width="784">




## Citation

If you find our work useful in your research, please consider starring the repo and citing:

```Bibtex
@inproceedings{kumar2024seabird,
   title={{SeaBird: Segmentation in Bird's View with Dice Loss Improves Monocular $3$D Detection of Large Objects}},
   author={Kumar, Abhinav and Guo, Yuliang and Huang, Xinyu and Ren, Liu and Liu, Xiaoming},
   booktitle={CVPR},
   year={2024}
}
```

## Single Camera (KITTI-360) Models

See [PanopticBEV](PanopticBEV)

## Multi-Camera (nuScenes) Models

See [HoP](HoP)

## Acknowledgements
We thank the authors of the following awesome codebases:
- [PanopticBEV](https://github.com/robot-learning-freiburg/PanopticBEV)
- [BBAVectors](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection) 
- [DEVIANT](https://github.com/abhi1kumar/DEVIANT.git)
- [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit)
- [HoP](https://github.com/Sense-X/HoP)

Please also consider citing them.

## Contributions
We welcome contributions to the SeaBird repo. Feel free to raise a pull request.

### &#8627; Stargazers
[![Stargazers repo roster for @nastyox/Repo-Roster](https://reporoster.com/stars/abhi1kumar/SeaBird)](https://github.com/abhi1kumar/SeaBird/stargazers)

### &#8627; Forkers
[![Forkers repo roster for @nastyox/Repo-Roster](https://reporoster.com/forks/abhi1kumar/SeaBird)](https://github.com/abhi1kumar/SeaBird/network/members)


## License
SeaBird code is under the [MIT license](https://opensource.org/license/mit).

## Contact
For questions, feel free to post here or drop an email to this address- ```abhinav3663@gmail.com```
