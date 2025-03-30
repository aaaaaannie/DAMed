## Current Models and Methods
- Baseline (SAMed for segment / MedFair for classification)
- Baseline + Fairness Loss
- FEBS
- FEBS + Fairness Loss
- GroupDRO
- GroupDRO + Fairness Loss
- VREx
- VREx + Fairness Loss

## Training Scripts
To train a specific model, run any of the following commands:
'''
python train-baseline.py                ## Train the baseline model

python train-baseline+fairness.py       ## Train the baseline model + our proposed fairness loss

python train-febs.py                    ## Train the FEBS model

python train-febs+fairness.py           ## Train the FEBS model + our proposed fairness loss

python train-groupDRO.py                ## Train the groupDRO model

python train-groupDRO+fairness.py       ## Train the groupDRO model + our proposed fairness loss

python train-VREx.py                    ## Train the VREx model

python train-VREx+fairness.py           ## Train the VREx model + our proposed fairness loss
'''

## multiple seeds

The`train.sh`script automates the process of training and testing these models with multiple random seeds. 
'''
bash train.sh
'''
