# Cryoto_predictor

# High Frequency Cryptocurrency Trading Signals Forecas -- CHAOYU DENG

## User GuideLines:
A detailed usage guide can be found in the Charpter: User Guide of the paper, and here are some basic usage instructions.

## Environment Configuration:
see requirements.txt

## Data Configuration:
see ./codes/data/guideline.txt

## Run Experiments:

Defult Model: LSTM lr:0.0005 lr-decay: 0.8 batch_size: 512 loss: Dice loss with alpha = 0.01
Defult Dateset: BTC-50

        python train_model.py


