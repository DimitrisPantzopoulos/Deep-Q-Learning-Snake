# Deep-Q-Learning-Snake
This project implements Deep Q-Learning from scratch to make an Agent to optimally play snake.

## Setup

Install the Python dependencies (tested with Python 3.13):

```shell
python -m pip install -r requirements.txt
```

Start training with:

```shell
python train.py
```

Evaluate the best checkpoint over 100 episodes:

```shell
python evaluate.py
```

Choose a checkpoint and episode count, or enable rendering:

```shell
python evaluate.py --checkpoint checkpoints/checkpoint_8000.pt --episodes 25 --render
```
