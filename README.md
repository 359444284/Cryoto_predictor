# Cryoto_predictor

# High Frequency Cryptocurrency Trading Signals Forecas

## Environment Configuration:
see requirements.txt

## Data Configuration:
see ./codes/data/guideline.txt

## Models Used in Essay

Defult Model: Bert_lastClsSep lr:5e-6 elr 0.9 batch size 4 loss DI 0.3 + MTD + ema + clamp + R-drop seed:15 (val: 0.5764    0.6633    0.6168)

        python train_model.py



Other Model:
1. Bert_lastClsSep lr:3e-6 elr 0.95 batch size 4 loss DI 0.3  + MTD + ema + FGM seed:15 （val: 0.6510 0.6020 0.6265）

        python train_model.py --model_output cls_spe --lr 3e-6 --dice_fact 0.3 --sample_num 2.5 --epochs 15 --atk FGM --ema True --grad_clamp False

2. deberta-v3-large + MTD + lstm_gru lr:5e-6, elr:0.9, batch size:4 loss:DI 0.5 3N seed:12 （val: 0.6630 0.6030 0.6316）

        python train_model.py --model_output lstm_gru --lr 5e-6 --dice_fact 0.5 --sample_num 3 --seed 12 --ema False --grad_clamp False --r_drop False

## Hyper Parameters And Defluat Value
You can add --+[Parameters] + value to set up you own Hyper Parameters.
- batch_size 4
- epochs 12
- lr 5e-6
- seed 15
- model microsoft/deberta-v3-large
- model_output cls_spe
- dice_fact 0.3
- atk ''
- ema True
- grad_clamp True
- r_drop True
- sample_num 2.5

## Result
python predict.py  (get the submission file and validation result)
