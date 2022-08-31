# Cryoto_predictor

# High Frequency Cryptocurrency Trading Signals Forecast

## User GuideLines:
A detailed usage guide can be found in the UserGuide.pdf and here are some basic usage instructions.



## Environment Configuration:
see requirements.txt

## Data Configuration:
see ./codes/data/guideline.txt

## Run Experiments:
        cd codes

Defult Model: DeepLOB lr:0.01 lr-decay: 0.8 batch_size: 512 loss: Dice loss with alpha = 0.5

Defult Dateset: FI-2010

Begin Train and Evaluation:
        
        python main.py
        
Evaluation:

        python predict.py

Cross-Validation For Our Cryptocurrency Dataset

        python cross_validation.py



