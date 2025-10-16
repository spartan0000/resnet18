# Resnet 18 \(and 34\) with the CIFAR-10 dataset
![](images/ResNet18image.png)


## Slight modification due to small size of the images (32 x 32)
 - Initial layer has a 3x3 kernel instead of 7x7
 - Stride of 1 and padding of 1 maintains spatial dimensions
 

### I've been tuning the model and experimenting with various hyperparameters and am  updating this README with the best performing model and its hyperparameters

### Transforms - random crop with padding of 4, random horizontal flip, to tensor, then normalization

- epochs - 60
- learning rate = 0.1 with cosine annealing with warm restarts with T_0 = 20
- batch size = 128
- optimizer - SGD with momentum 0.9
- loss function - cross entropy loss with label smoothing = 0.1
- device - cuda
- automatic mixed precision (amp) - with Grad Scaler
- weight decay = 1e-3  
- dropout - 0.5 just prior to the fully connected layer




### With this model I was able to get a test accuracy of 90.12%

### Experimenting with stochastic depth to try and further improve performance but Resnet 18 seems too shallow for this to work so will have to explore deeper Resnets

### Using Weight and Biases (wandb.ai) to track and log experiments - it's awesome!


### 10-15-2025: Added resnet34 as a command line argument so that I can run both resnet 18 and 34 by changing the --model argument
- epochs 70
- lr = 5e-2 | cosine annealing with warm restarts | T_0 = 10, T_mult = 2
- batch size = 128
- optimizer - sgd with momentum 0.9
- loss function - cross entropy loss with label smoothing 0.1
- weight decay = 5e-4
- amp with grad scaler
- drop out 0.5

### got a test accuracy of 91.49% with this version of resnet 34.  Training time of 50 mins on my GPU

### next step is to implement a simple stochastic depth with drop prob 0.2 and see what happens...

