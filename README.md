# Neural Network Models for MNIST Classification

This repository contains four neural network models designed to classify MNIST digits. The goal of this assignment is to reach an accuracy of 99.40 % (or more in continous epochs) within 15 epochs and less than 8000 parameters

## Overview
The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). The objective is to classify these images into the correct digit class. All models in this repository are implemented using PyTorch.

## Requirements
To run the models, ensure the following dependencies are installed:
- Python 3.7+
- PyTorch
- torchvision
- tqdm
- torchsummary

Install dependencies using:
```bash
pip install torch torchvision tqdm torchsummary
```

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Run any model notebook:
   ```bash
   jupyter notebook Model1.ipynb
   ```
3. Follow the instructions in the notebook to train and test the model.


## Models

### Model 1
 **Logs**:
 ```python
 EPOCH: 0
Loss=0.0803607851266861 Batch_id=468 Accuracy=91.00: 100%|██████████| 469/469 [00:20<00:00, 22.92it/s]

Test set: Average loss: 0.0673, Accuracy: 9805/10000 (98.05%)

EPOCH: 1
Loss=0.07429290562868118 Batch_id=468 Accuracy=98.20: 100%|██████████| 469/469 [00:17<00:00, 26.08it/s]

Test set: Average loss: 0.0509, Accuracy: 9847/10000 (98.47%)

EPOCH: 2
Loss=0.022153906524181366 Batch_id=468 Accuracy=98.51: 100%|██████████| 469/469 [00:16<00:00, 28.33it/s]

Test set: Average loss: 0.0359, Accuracy: 9894/10000 (98.94%)

EPOCH: 3
Loss=0.03196047991514206 Batch_id=468 Accuracy=98.74: 100%|██████████| 469/469 [00:16<00:00, 28.37it/s]

Test set: Average loss: 0.0301, Accuracy: 9917/10000 (99.17%)

EPOCH: 4
Loss=0.0514090396463871 Batch_id=468 Accuracy=98.92: 100%|██████████| 469/469 [00:16<00:00, 29.08it/s]

Test set: Average loss: 0.0373, Accuracy: 9884/10000 (98.84%)

EPOCH: 5
Loss=0.01636817865073681 Batch_id=468 Accuracy=98.98: 100%|██████████| 469/469 [00:16<00:00, 27.66it/s]

Test set: Average loss: 0.0277, Accuracy: 9918/10000 (99.18%)

EPOCH: 6
Loss=0.07352645695209503 Batch_id=468 Accuracy=98.99: 100%|██████████| 469/469 [00:17<00:00, 27.50it/s]

Test set: Average loss: 0.0221, Accuracy: 9938/10000 (99.38%)

EPOCH: 7
Loss=0.007313742768019438 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:16<00:00, 28.69it/s]

Test set: Average loss: 0.0250, Accuracy: 9927/10000 (99.27%)

EPOCH: 8
Loss=0.025066301226615906 Batch_id=468 Accuracy=99.10: 100%|██████████| 469/469 [00:16<00:00, 27.97it/s]

Test set: Average loss: 0.0229, Accuracy: 9926/10000 (99.26%)

EPOCH: 9
Loss=0.008187985047698021 Batch_id=468 Accuracy=99.19: 100%|██████████| 469/469 [00:16<00:00, 27.75it/s]

Test set: Average loss: 0.0241, Accuracy: 9930/10000 (99.30%)

EPOCH: 10
Loss=0.0016372008249163628 Batch_id=468 Accuracy=99.27: 100%|██████████| 469/469 [00:16<00:00, 28.65it/s]

Test set: Average loss: 0.0220, Accuracy: 9929/10000 (99.29%)

EPOCH: 11
Loss=0.029993591830134392 Batch_id=468 Accuracy=99.31: 100%|██████████| 469/469 [00:16<00:00, 28.86it/s]

Test set: Average loss: 0.0212, Accuracy: 9933/10000 (99.33%)

EPOCH: 12
Loss=0.026637496426701546 Batch_id=468 Accuracy=99.28: 100%|██████████| 469/469 [00:16<00:00, 28.47it/s]

Test set: Average loss: 0.0218, Accuracy: 9936/10000 (99.36%)

EPOCH: 13
Loss=0.006130168214440346 Batch_id=468 Accuracy=99.33: 100%|██████████| 469/469 [00:17<00:00, 27.49it/s]

Test set: Average loss: 0.0266, Accuracy: 9920/10000 (99.20%)

EPOCH: 14
Loss=0.001742902328260243 Batch_id=468 Accuracy=99.33: 100%|██████████| 469/469 [00:16<00:00, 28.71it/s]

Test set: Average loss: 0.0232, Accuracy: 9929/10000 (99.29%)
```

**Target:**

Total parameter count should be within 8000

Epochs 15 or less

Reach test accuracy greater than or equal to 99.40 in continous epoch

**Results:**

Parameters: 8,272

Best Training Accuracy: 99.33

Best Test Accuracy: 99.38

**Analysis:**

Model has been made lighter

Model is not too overfitting but there is minor inconsistency in test performance in later epochs

Required accuracy has not been reached

**What can be done:**

Introduce image augmentation to introduce variability

Increase the dropout rate to reduce minor overfitting.



### Model 2

**Logs**:
 ```python
 EPOCH: 0
Loss=0.07724960893392563 Batch_id=468 Accuracy=91.64: 100%|██████████| 469/469 [00:32<00:00, 14.60it/s]

Test set: Average loss: 0.0671, Accuracy: 9799/10000 (97.99%)

EPOCH: 1
Loss=0.026071548461914062 Batch_id=468 Accuracy=98.02: 100%|██████████| 469/469 [00:36<00:00, 12.97it/s]

Test set: Average loss: 0.0478, Accuracy: 9858/10000 (98.58%)

EPOCH: 2
Loss=0.08313081413507462 Batch_id=468 Accuracy=98.41: 100%|██████████| 469/469 [00:32<00:00, 14.52it/s]

Test set: Average loss: 0.0479, Accuracy: 9828/10000 (98.28%)

EPOCH: 3
Loss=0.04827054217457771 Batch_id=468 Accuracy=98.66: 100%|██████████| 469/469 [00:31<00:00, 14.86it/s]

Test set: Average loss: 0.0415, Accuracy: 9870/10000 (98.70%)

EPOCH: 4
Loss=0.024755120277404785 Batch_id=468 Accuracy=98.69: 100%|██████████| 469/469 [00:32<00:00, 14.36it/s]

Test set: Average loss: 0.0331, Accuracy: 9893/10000 (98.93%)

EPOCH: 5
Loss=0.010424304753541946 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:31<00:00, 14.93it/s]

Test set: Average loss: 0.0312, Accuracy: 9908/10000 (99.08%)

EPOCH: 6
Loss=0.03885653242468834 Batch_id=468 Accuracy=98.96: 100%|██████████| 469/469 [00:31<00:00, 15.11it/s]

Test set: Average loss: 0.0315, Accuracy: 9896/10000 (98.96%)

EPOCH: 7
Loss=0.009951157495379448 Batch_id=468 Accuracy=99.01: 100%|██████████| 469/469 [00:31<00:00, 14.99it/s]

Test set: Average loss: 0.0294, Accuracy: 9914/10000 (99.14%)

EPOCH: 8
Loss=0.12311068922281265 Batch_id=468 Accuracy=99.06: 100%|██████████| 469/469 [00:31<00:00, 14.97it/s]

Test set: Average loss: 0.0247, Accuracy: 9919/10000 (99.19%)

EPOCH: 9
Loss=0.055809687823057175 Batch_id=468 Accuracy=99.08: 100%|██████████| 469/469 [00:31<00:00, 14.75it/s]

Test set: Average loss: 0.0240, Accuracy: 9925/10000 (99.25%)

EPOCH: 10
Loss=0.038612861186265945 Batch_id=468 Accuracy=99.14: 100%|██████████| 469/469 [00:31<00:00, 14.68it/s]

Test set: Average loss: 0.0230, Accuracy: 9919/10000 (99.19%)

EPOCH: 11
Loss=0.009483086876571178 Batch_id=468 Accuracy=99.20: 100%|██████████| 469/469 [00:33<00:00, 13.95it/s]

Test set: Average loss: 0.0258, Accuracy: 9919/10000 (99.19%)

EPOCH: 12
Loss=0.015231919474899769 Batch_id=468 Accuracy=99.19: 100%|██████████| 469/469 [00:31<00:00, 15.09it/s]

Test set: Average loss: 0.0251, Accuracy: 9922/10000 (99.22%)

EPOCH: 13
Loss=0.013160977512598038 Batch_id=468 Accuracy=99.25: 100%|██████████| 469/469 [00:31<00:00, 14.99it/s]

Test set: Average loss: 0.0287, Accuracy: 9917/10000 (99.17%)

EPOCH: 14
Loss=0.005055147223174572 Batch_id=468 Accuracy=99.25: 100%|██████████| 469/469 [00:32<00:00, 14.47it/s]

Test set: Average loss: 0.0215, Accuracy: 9927/10000 (99.27%)

 ```

**Target:**

Reduce the parameter count

Keep the epochs within 15

Reach test accuracy greater than 99.4

**Results:**

Parameters: 6,778

Best Training Accuracy: 99.25

Best Test Accuracy: 99.27

**Analysis:**

Parameter count has been reduced

The model has been significantly optimized, achieving a much lighter architecture.

Accuracy of 99.4% has not been reached test accuracy peaks around 99.27%.

**What can be done**

Introduce ReduceLROnPlateau scheduler for dynamically updating the learning rate

Try using Adam optimizer to stabilize learning

Gradually increase the number of feature maps in intermediate layers but still keep the count under 8000


### Model 3
**Logs**:
 ```python
/opt/conda/envs/pytorch/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
EPOCH: 0
Loss=0.06316276639699936 Batch_id=468 Accuracy=95.75: 100%|██████████| 469/469 [00:08<00:00, 58.35it/s] 

Test set: Average loss: 0.0532, Accuracy: 9816/10000 (98.16%)

EPOCH: 1
Loss=0.01732787676155567 Batch_id=468 Accuracy=98.31: 100%|██████████| 469/469 [00:07<00:00, 60.94it/s]  

Test set: Average loss: 0.0546, Accuracy: 9834/10000 (98.34%)

EPOCH: 2
Loss=0.09690334647893906 Batch_id=468 Accuracy=98.58: 100%|██████████| 469/469 [00:07<00:00, 60.81it/s]  

Test set: Average loss: 0.0320, Accuracy: 9904/10000 (99.04%)

EPOCH: 3
Loss=0.04262491688132286 Batch_id=468 Accuracy=98.72: 100%|██████████| 469/469 [00:07<00:00, 61.81it/s]  

Test set: Average loss: 0.0501, Accuracy: 9850/10000 (98.50%)

EPOCH: 4
Loss=0.015266002155840397 Batch_id=468 Accuracy=98.90: 100%|██████████| 469/469 [00:07<00:00, 62.19it/s] 

Test set: Average loss: 0.0317, Accuracy: 9903/10000 (99.03%)

EPOCH: 5
Loss=0.04794388636946678 Batch_id=468 Accuracy=98.91: 100%|██████████| 469/469 [00:07<00:00, 61.35it/s]  

Test set: Average loss: 0.0310, Accuracy: 9908/10000 (99.08%)

EPOCH: 6
Loss=0.02243087626993656 Batch_id=468 Accuracy=99.00: 100%|██████████| 469/469 [00:07<00:00, 60.91it/s]  

Test set: Average loss: 0.0251, Accuracy: 9918/10000 (99.18%)

EPOCH: 7
Loss=0.022157451137900352 Batch_id=468 Accuracy=99.00: 100%|██████████| 469/469 [00:07<00:00, 62.73it/s] 

Test set: Average loss: 0.0315, Accuracy: 9895/10000 (98.95%)

EPOCH: 8
Loss=0.029119694605469704 Batch_id=468 Accuracy=99.11: 100%|██████████| 469/469 [00:07<00:00, 61.65it/s] 

Test set: Average loss: 0.0396, Accuracy: 9877/10000 (98.77%)

EPOCH: 9
Loss=0.04288138821721077 Batch_id=468 Accuracy=99.13: 100%|██████████| 469/469 [00:07<00:00, 61.64it/s]   

Test set: Average loss: 0.0298, Accuracy: 9909/10000 (99.09%)

EPOCH: 10
Loss=0.00682647293433547 Batch_id=468 Accuracy=99.52: 100%|██████████| 469/469 [00:07<00:00, 61.36it/s]   

Test set: Average loss: 0.0159, Accuracy: 9944/10000 (99.44%)

EPOCH: 11
Loss=0.000698699674103409 Batch_id=468 Accuracy=99.62: 100%|██████████| 469/469 [00:07<00:00, 62.78it/s]  

Test set: Average loss: 0.0174, Accuracy: 9944/10000 (99.44%)

EPOCH: 12
Loss=0.007735205814242363 Batch_id=468 Accuracy=99.66: 100%|██████████| 469/469 [00:07<00:00, 62.28it/s]  

Test set: Average loss: 0.0149, Accuracy: 9956/10000 (99.56%)

EPOCH: 13
Loss=0.031765781342983246 Batch_id=468 Accuracy=99.67: 100%|██████████| 469/469 [00:07<00:00, 62.25it/s]  

Test set: Average loss: 0.0163, Accuracy: 9946/10000 (99.46%)

EPOCH: 14
Loss=0.02382434904575348 Batch_id=468 Accuracy=99.71: 100%|██████████| 469/469 [00:07<00:00, 62.00it/s]   

Test set: Average loss: 0.0158, Accuracy: 9955/10000 (99.55%)
```
**Target:**

Introduce Scheduler

Keep the epochs within 15

Reach test accuracy greater than 99.4

**Results:**

Parameters: 7,510

Best Training Accuracy: 99.71

Best Test Accuracy: 99.56

**Analysis:**

Parameters have been successfully constrained under 8000

The model shows  clear improvement in test accuracy reaching 99.44% by the 10th epoch and stabilizing at 99.55% by the end of training

The scheduler allowed the model to generalize better and reach higher accuracy


## Evaluation

- **Input**:  28*28 - grayscale images
- **Optimizer**:  SGD with momentum for first 2 models and Adam for the 3rd model
- **Loss Function**: Negative Log Likelihood
- **Epochs**: 15
- **Batch Size**: 128 (when CUDA is available)


## Models Performance


| Model     | Parameters | Best Training Accuracy | Best Test Accuracy  |
|-----------|------------|------------------------|---------------------|
| Model 1   | 8,272      | 99.33%                 | 99.38%              |
| Model 2   | 6,778      | 99.25%                 | 99.27%              |
| Model 3   | 7,510      | 99.71%                 | 99.56%              |



### Key Findings:

- **Best Model**: Model 3 achieved the highest test accuracy (99.56%).
- **Key Improvement**: Introducing a learning rate scheduler in Model 3 helped in stabilizing training and improving test accuracy.
- **Target Achievement**: The accuracy target of 99.4% was reached in Model 3.
