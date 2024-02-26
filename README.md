# World Models with Augmented Replay


In this project, we added replay to a world model architecture for continual RL. The project was based on the [RFR](https://wba-initiative.org/en/21935/), and the first version was conducted as a Masters project at Monash University.

See the preprint on [arXiv](https://arxiv.org/abs/2401.16650).

**Abstract**

When the environments of a reinforcement learning problem undergo changes, the learning algorithm must appropriately balance the two potentially conflicting criteria of \textit{stability} and \textit{plasticity}. Such a scenario is known as continual learning, and a successful system should accommodate challenges in retaining agent performance on already learned tasks (stability), whilst being able to learn new tasks (plasticity).
  The first-in-first-out buffer is commonly used to enhance learning in such settings but demands significant memory requirements. We explore the application of an augmentation to this buffer which alleviates the memory constraints, and use it with a world model model-based reinforcement learning algorithm to evaluate its effectiveness in facilitating continual learning.
  We evaluate the effectiveness of our method in Procgen and Atari reinforcement learning benchmarks and show that the distribution matching augmentation to the replay buffer used in the context of latent world models can successfully prevent catastrophic forgetting with significantly reduced computational overhead.
  Yet, we also find such a solution to not be entirely infallible, and other failure modes such as the opposite --- lacking plasticity and being unable to learn a new task to be a potential limitation in continual learning systems.

## Requirements
- Python (3.10)
- PyTorch

Before you run experiments, you will have to install dependencies listed inside `requirements.txt`

You can install the dependencies using `pip install -r requirements.txt`.

## Getting Started

Run experiments with ```python train.py```

There is an optional parameter to set a config.
A sample configuration file can be generated with the example in train.py, and exporting as json.
