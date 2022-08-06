## File structure
- pytorch-cifar
    - resnet implementation with VAECNN expriments as well
- pytorch-adversarial-training
    - adversarial training repor
- pytorch-vae
    - vae implementation in pytorch
- argument.py
    - include the input arguments for example learning rate etc
- adversarial_training_main.py
    - my implementation for adversarial training
- fast_gradient_sign_untargeted.py
    - FGSM attack method
- VAECNN_main.py
    - my implementation for VAE CNN

## Commonly used commands
For adversarial training 
```
$ python3 adversarial_training_main.py --data_root . --batch_size=64 --learning_rate=0.05
```

To train the VAECNN model
```
$ python3 VAECNN_main.py --saved_file_name=vaeFirstLayerChangedklckpt.pth
```


## log

14 Jul 2022
- implement it to accept model name
- haven't executed on HPC yet