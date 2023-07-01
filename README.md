# App Manual
### Libs you will need to run this app locally.
* pygame
* pytorch
* numpy
### Implementation of your own model.
1) Your model must take tensor of (1, 1, 28, 28) shape and return tensor with 10 scores for each digit from 0 to 9. 
If your model is different, you should change `preproc` function in `preprocessor.py`.
2) You should add trained model file to project directory and change content of `model.py` so your model would work but not mine.
#### That's all you need to know, have fun!
