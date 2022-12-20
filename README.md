# Dynamic Magnetic Resonance Imaging via Unfolded Algorithm

## Description

This is a git repository of King BUT project entitled **Unfolded L+S image reconstrucion for DCE-MRI data**. The package is capable of training the unfolded algorithm and show the results on testing dataset. 

For the code to workm, some configuration has to be done inside the file `default_config.json` (or creating a `custom_config.json` and editing that). The path to data must be set in `datafold` parameter, there it is expected that two folder exist (`train`, `test`).

More detailed configuration, i.e. regularization parametrization can be also changed in this file.

Then it is possible to run the training and testing via a script (`/CP_pytorch/reconstruction.py`), the process is acompanied by status info in the console. Results are then saved in the data folder in a new timestamped subfolder alongside the configuration used for the particular run.


### Dataset folder structuer
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

## Data avalibility

As the data are quite large, they can be obtained on request from the project coordinator Ondřej Mokrý (xmokry12@vut.cz).

## Acknowledgement

The work was supported by the Grant Dynamic Magnetic Resonance Imaging via Unrolled Algorithm (FEKT-K-22-7772) realised within the project Quality Internal Grants of BUT (KInG BUT), Reg. No. CZ.02.2.69/0.0/0.0/19\_073/0016948, which is financed from the OP RDE.

## Citation

Should you use some of our results please cite as:

MOKRÝ, O.; VITOUŠ, J. Unrolled L+S decomposition for compressed sensing in magnetic resonance imaging. Elektrorevue - Internetový časopis (http://www.elektrorevue.cz), 2022, vol. 24, no. 3, p. 86-93. ISSN: 1213-1539.





