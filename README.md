# Dynamic Magnetic Resonance Imaging via Unfolded Algorithm

## Description

This is a git repository of King BUT project regarding Unfolded L+S image reconstrucion for DCE-MRI data. The algortihm will train the algorithm and show the results on testing dataset. 

For the code to work configuration has to be done inside the `default_config.json` (or creating a `custom_config.json` and editing that). The path do data must be set in `datafold` parameter, there it is expected to exist two folders (`train, test`)

Further configuration, i.e. regularization parametrization can be also changed in this file.

Then it is possible to run the training and testing via a script (`/CP_pytorch/reconstruction.py`), the process is acompanied by status info in the console. Results are then saved in the Dataset folder in a new timestamped folder alongside used configuration.


### Dataset folder structuer
```
datafold
|
└───train
|    |
|    └─ (training dataset)
|
└───test
|     |
|     └─ (testing dataset)
|
└─reco_XXXXXXXX(reconstructed data)
```

## Data avalibility

As the data are quite large, they can be obtained on request from the project coordinator Ondrej Mokry (xmokry12@vut.cz)

## Acknowledgement

This project was supported by Student grant FEKT-K-22-7772 as part of OP VVV CZ.02.2.69/0.0/0.0/19_073/0016948. 

## Citation

Should you use some of our results please cite as:

MOKRÝ, O.; VITOUŠ, J. Unrolled L+S decomposition for compressed sensing in magnetic resonance imaging. Elektrorevue - Internetový časopis (http://www.elektrorevue.cz), 2022, vol. 24, no. 3, p. 86-93. ISSN: 1213-1539.





