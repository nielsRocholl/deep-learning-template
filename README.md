# Deep Learning Project Template
by Niels Rocholl

This repository serves as a structured template designed to streamline the setup of Deep Learning projects. It encapsulates a range of fundamental utilities that are commonly required throughout the developmental phases of a DL project. The key features included in this template are as follows:

- **Configuration Management**: Utilizes YAML files for configuration management.
- **Experiment Tracking**: Incorporates Weights and Biases for precise experiment tracking.
- **Configuration Loaders and Writers**: Facilitates easy management of configuration data.
- **Custom Logger**: A logger designed to support distributed computing environments.
- **Custom Argument Parser**: A argument parser to streamline command-line interactions.
- **Class Registry**: A registry to effortlessly map strings to classes.
- **Dataset Class**: A dataset class to handle data operations seamlessly.
- **IO Class**: A dedicated IO class to manage data input and error handling efficiently.

Each function within this template is documented with comprehensive docstrings, comments, and proper typing. The primary adjustment required post-setup is updating the Weights and Biases Project name within the `main.py` file. In addition to example text in the configuration files. 

The project structure laid out by this template is illustrated below:

```plaintext
toy-problem/
│
├── DATASET.md
├── README.md
├── cfgs
│   ├── dataset_cfgs
│   │   └── part_net.yaml
│   └── train.yaml
│
├── dataset
│   ├── build.py
│   ├── io.py
│   └── part_net.py
│
├── experiments
│   └── train
│       └── cfgs
│           ├── TFBoard
│           │   └── experiment
│           └── experiment
│               └── wandb
│
├── main.py
├── models
│
├── output
│   ├── figures
│   └── trained-models
│
├── requirements.txt
├── tools
│   ├── model_tester.py
│   └── model_trainer.py
│
└── utils
    ├── config.py
    ├── logger.py
    ├── misc.py
    ├── parser.py
    └── registry.py
```
