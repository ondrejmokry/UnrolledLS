# Dynamic Magnetic Resonance Imaging via Unfolded Algorithm

## Description

This is a git repository of King BUT project entitled **Unfolded L+S image reconstruction for DCE-MRI data**. The package is capable of training the unfolded algorithm and show the results on testing dataset. 

For the code to work, some configuration has to be done inside the file `default_config.json` (or creating a `custom_config.json` and editing that). The path to data must be set in `datafold` parameter, there it is expected that two folder exist (`train`, `test`).

More detailed configuration, i.e. regularization parametrization can be also changed in this file.

Then it is possible to run the training and testing via a script (`/CP_pytorch/reconstruction.py`), the process is accompanied by status info in the console. Results are then saved in the data folder in a new timestamped subfolder alongside the configuration used for the particular run.


### Dataset folder structure
```
datafold
|
└───train
|    |
|    └─ (training dataset)
|
└───test
|    |
|    └─ (testing dataset)
|
└─reco_XXXXXXXX(reconstructed data)
```

### Supplementary material

Besides the codes in `CP_pytorch`, the repository contains the following additional materials:
- a brief description of the derivation of the (unfolded) algorithm (the file `derivation.pdf`),
- a Matlab plotting script `animation_of_unfolded.m`, which allows to load files saved by the testing procedure, animate the sequence and plot perfusion curves.

## Data availability

As the data are quite large, they can be obtained on request from the project coordinator Ondřej Mokrý (xmokry12@vut.cz).

## Acknowledgement

The work was supported by the Grant Dynamic Magnetic Resonance Imaging via Unrolled Algorithm (FEKT-K-22-7772) realised within the project Quality Internal Grants of BUT (KInG BUT), Reg. No. CZ.02.2.69/0.0/0.0/19\_073/0016948, which is financed from the OP RDE.

## Citation

Should you use some of our results please cite the following:

MOKRÝ, O.; VITOUŠ, J. Unrolled L+S decomposition for compressed sensing in magnetic resonance imaging. Elektrorevue - Internetový časopis (http://www.elektrorevue.cz), 2022, vol. 24, no. 3, p. 86-93. ISSN: 1213-1539.

MOKRÝ, O.; VITOUŠ, J. Unfolded Low-rank + Sparse Reconstruction for MRI. In Proceedings II of the 28th Conference STUDENT EEICT 2022 Selected papers. 1. Brno: Brno University of Technology, Faculty of Electrical Engineering and Communication, 2022. p. 271-275. ISBN: 978-80-214-6030-0.
