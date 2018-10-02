# SCORES
This is an Pytorch visual demo of the paper "[SCORES: Shape Composition with Recursive Substructure Priors](https://kevinkaixu.net/projects/scores.html)". This is a neural network which learns structure fusion for 3D shape composition

## Usage
**Dependancy**

This implementation should be run with Python 3.x and Pytorch 0.4.0.

**Demo**
```
python testMerge.py
```
This will show some visual composition results based on our SCORES network. The input initial composition and the refined composition given by SOCRES would be displayed using the utility functions in draw3dobb.py provided in this project.

## Citation
If you use this code, please cite the following paper.
```
@article {zhu_siga18,
    title = {SCORES: Shape Composition with Recursive Substructure Priors},
    author = {Chenyang Zhu and Kai Xu and Siddhartha Chaudhuri and Renjiao Yi and Hao Zhang},
    journal = {ACM Transactions on Graphics (SIGGRAPH Asia 2018)},
    volume = {37},
    number = {6},
    pages = {to appear},
    year = {2018}
}
```
