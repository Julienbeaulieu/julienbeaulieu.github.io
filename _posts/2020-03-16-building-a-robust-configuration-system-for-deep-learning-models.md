
## Introduction

A few weeks ago, I joined a recurring deep learning meetup where a few experienced practitioners get together to work on different computer vision projects. Their latest endeavour was to build a pipeline in PyTorch to attempt a Kaggle competition. The competition was a multitask classification problem where the goal is to recognize different components of Bengali characters. Their approach was partly to get a good score for the competition, and mostly to build a customizable project structure and pipeline that could be reused for future deep learning projects, whether for work or personal. I loved their approach and decided to join them.

This post is an attempt to summarize key concepts that could be used for any machine learning project. Specifically, I’ll be covering three main topics: 
1. [Project organization and a polished directory structure](#a-polished-directory-structure)
2. [How to build a robust configuration system and experimentation framework](#setting-up-a-robust-configuration-system-and-experimentation-framework)
3. [How to customize our model's architecture based on our configurations](#customize-our-models-architecture-based-on-our-configuration)

To give some insight of the end result of part 3, we will be able to pass in any number of hyperparameters, settings, and modules to a configuration file and have a routine that assembles the model correctly given these configurations when running our training function. I walk through all of the steps required to achieve this.  

All the code for this project can be found [here](https://github.com/Julienbeaulieu/kaggle-computer-vision-competition). The group effort is [here](https://gitlab.com/cvnnig/bengali.ai). 

Credits: I did not write this architecture myself. I did however familiarize myself with it, rebuilt it from scratch, and contributed a bit along the way. Credit goes to my *\*wishes to stay anonymus*\* meeting partner, and to [Yang](https://www.linkedin.com/in/dryangding/), who helped standardize & automate a lot of the work and also organizes the meetups. This post was also inspired by the [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure.

## A Polished Directory Structure

While creating a state-of-the-art model and implementing recent papers is critical, there is a tendency to negate the quality and the structure of the code that creates these models. Data science code should be about flexibility, reproducibility and being able to experiment fast by testing different approaches. This is essential if ever we need to run some models again in the future, if we’re onboarding a new data scientist, or if we wish to standardize the way models are built throughout the organization. A good project starts with a well-thought directory structure. I will break down how we approached this in three parts.

### 1. Data organization

We’re simplifying things a little and imagining that the data engineers have already collected the data for us to experiment with.

This is how the project’s data is organized:

    ├── README.md          
    ├── data
    │   ├── external        <- Data from third party sources.
    │   ├── interim         <- Intermediate data that has been transformed.
    │   ├── processed       <- The final, canonical data sets for modeling.
    │   └── raw             <- The original, immutable data dump.
    │

We first put the original data in a folder called `raw`. The transformed data used for our model will reside in `interim` and `processed` folders. Any external data should be kept separate as well, in `external`.

### 2. Experiments

We need a place that allows us to manage the experiment process of evaluating multiple models and ideas.

    ├── notebooks           <- Jupyter notebooks for exploration and communication
    │
    ├── experiments         <- Trained models, predictions, experiment configs, Tensorboard logs, backups
    │   ├─ exp01
    │   │  ├── model_backups
    │   │  ├── results
    │   │  └── tensorboard logs
    │   │
    │   ├─ exp02
    │   ...
    │

`Notebooks` folder contains all the notebooks used for exploring, experimenting and visualization during the initial phases of the project.

`Experiments` is where we store the trained models, model backups, as well as the results and configurations for a given experiment. We use a YAML config file for each experiment, something I'll detail later.

### 3.  Source code of our model

Let’s have a look at the actual code structure required to build the model.

    src                              
    ├── config
    │   ├── config.py                <- Default configs for the model 
    │   └── experiments
    │       ├── exp01_config.yaml    <- Configs for a specific experiment. Overwrites default configs
    │       └── exp02_config.yaml
    │       xz
    ├── data                  
    │   ├── make_dataset.py          <- Script to generate data
    │   ├── bengali_data.py          <- Custom Pytorch Dataset, DataLoader & Collator class
    │   └── preprocessing.py         <- Custom data augmentation class
    │
    ├── modeling                                  
    │   ├── backbone                 <- Model backbone architecture
    │   │   ├── se_resnext50.py
    │   │   └── densenet121.py
    │   │
    │   ├── layers                   <- Custom layers
    │   │   └── linear.py
    │   │
    │   ├── meta_arch                <- Scripts to combine backbone + head
    │   │   ├── baseline.py
    │   │   └── build.py
    │   │
    │   ├── head                     <- Build the head of the model
    │   │   ├── build.py
    │   │   └── simple_head.py
    │   │
    │   └── solver                   <- Scripts for building loss function, evaluation and optimizer
    │       ├── loss
    │       │   ├── build.py
    │       │   ├── softmax_cross_entropy.py
    │       │   └── label_smoothing_ce.py
    │       ├── evaluation.py
    │       └── optimizer.py 
    │ 
    ├── tools                        <- Training loop and custom helper functions 
    │   ├── train.py
    │   └── registry.py 
    │ 
    └── visualization                <- Scripts for exploratory results & visualizations 
           └── visualize.py

Well organized code tends to be self-documenting in that the organization itself provides context for the code without much overhead.

This may look complicated at first glance but it's relativelty straight forward once we break it down:

- `config`: More details in section 2 but essentially, this is where we store our default and experiment configurations.

- `data`: Contains scripts and modules to manage the loading and construction of our dataset. These scripts handle data reading from disk, transforming the raw data, building a custom Pytorch Dataset, Dataloader and Collator class, as well as all of the data augmentations required for our task.

- `modeling`: Includes all the necessary modules in order to build a model. Different parts of the model are constructed separately and assembled in `meta_arch` scripts. For example, `backbone` defines the neural network architectures used. Here, only the computational graph is defined. These objects are agnostic to the input and output shapes, model losses, and training methodology. `solver` contains modules to build our loss function(s), metric evaluation pipeline, as well as our optimizer and scheduler.

- `tools`: Contains our training loop that trains a model with respect to a set of model configurations. This code interacts with the optimizer, the scheduler, the loss function, and handles logging during training. This folder is also where we also store all our helper functions.

- `visualization`: Run scripts to create any visuals we’ll want to use for a report or to better understand our dataset.

Our data is now organized and tidy. 


## Setting up a Robust Configuration System and Experimentation Framework

In this section I show how to setup a system that lets you easily change, save and iterate on as many model hyperparameters and architecture/module choices as you want thanks to a well designed configuration system.

As noted by the [paper on hidden technical debt in machine learning systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems.pdf), setting up a proper configuration framework is essential yet often overlooked.

> “Another potentially surprising area where debt can accumulate is in the configuration of machine learning systems. Any large system has a wide range of configurable options, including which features are used, how data is selected, a wide variety of algorithm-specific learning settings, potential pre- or post-processing, verification methods, etc. We have observed that both researchers and engineers may treat configuration (and extension of configuration) as an afterthought. Indeed, verification or testing of configurations may not even be seen as important”

In light of this, we propose a way to organize, store, and run experiments efficiently. Here are the steps:

### Step 1. Keeping secrets and configuration out of version control

Since we really don't want to leak our AWS/GCP secret key or Postgres username and password on Github, we have to be mindful of how we pass that information into our model.

Therefore, we store our secret config variables in a special file: `.env`. We use a package called [python-dotenv](https://github.com/theskumar/python-dotenv) to load these variables automatically. It allows us to load up all the entries in the `.env` file as environment variables and are accessible with `os.environ.get`.

We first have to create a `.env` file in the project root folder. Adding it to `.gitignore` will ensure it never gets committed into the version control repository. Here's an example of how to use the library (adapted from the library documentation):

```python
import os
from dotenv import load_dotenv, find_dotenv

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables*
load_dotenv(dotenv_path)

PATH_DATA_RAW = Path(os.getenv("PATH_DATA_RAW"))
GCP_KEY = Path(os.getenv("GCP_KEY"))
```

We now have access to these secret variables and will use them in our configurations. 

### Step 2: Use YACS to define all default model configurations in one file

To achieve this we use a library called YACS, short for Yet Another Configuration System. It was created out of experimental configuration systems used in py-faster-rnn and Detectron by Facebook. It helps define and manage system configurations such as hyperparameters and architecture/module choices for training a model. A tool like this one is essential to reproducibility and is a fundamental component of the system.

If we take a closer look at our `config` folder mentioned above, it is comprised of a default `config.py` file which uses YACS. Here is a snippet of what it looks like:

```python
from yacs.config import CfgNode as ConfigurationNode

# YACS overwrite these settings using YAML
__C = ConfigurationNode()

# data augmentation parameters with albumentations library
__C.DATASET.AUGMENTATION = ConfigurationNode()
__C.DATASET.AUGMENTATION.BLURRING_PROB = 0.25
__C.DATASET.AUGMENTATION.GAUSS_NOISE_PROB = 0.25
__C.DATASET.AUGMENTATION.GAUSS_VAR_LIMIT =(10.0, 40.0)
__C.DATASET.AUGMENTATION.BLUR_LIMIT = 7
...

# model backbone configs
__C.MODEL.BACKBONE = ConfigurationNode()
__C.MODEL.BACKBONE.NAME = 'mobilenet_v2' 
__C.MODEL.BACKBONE.RGB = True
__C.MODEL.BACKBONE.PRETRAINED_PATH = 'C:/data-science/kaggle/bengali.ai/models/mobilenet_v2-b0353104.pth'

# model head configs
__C.MODEL.HEAD = ConfigurationNode()
__C.MODEL.HEAD.NAME = 'simple_head_module'
__C.MODEL.HEAD.ACTIVATION = 'leaky_relu'
__C.MODEL.HEAD.OUTPUT_DIMS = [168, 11, 7]
__C.MODEL.HEAD.INPUT_DIM = 1280  # mobilenet_v2 
__C.MODEL.HEAD.HIDDEN_DIMS = [512, 256]
__C.MODEL.HEAD.BATCH_NORM = True
__C.MODEL.HEAD.DROPOUT = 0.4
...
```

As you can see, all hyperparameters as well as certain modules (i.e ‘leaky_relu’, ‘simple_head’, ‘SE_ResNeXT50’, etc) are all defined in this file.

### Sidebar

There is a very important detail to note: while it is very straightforward to handle hyperparameter configuration – simply by passing a float, a list, or a bool where appropriate, then plug and play inside our model - module configuration requires more work. That is because passing a module name such as ‘simple_head_module’ in the config must trigger the execution of a Python module when it is called and build the model appropriatly. How is this achieved? The short answer is: by creating a Registry helper class for managing these modules as well as function decorators. This is the topic of part 3 of this article. 

### End sidebar

### Step 3: Define our experiment configurations

In the `config` folder, we also have a several YAML configs files. Each YAML file corresponds to one experiement that we've created. Here is an example:

```yaml
DATASET:
  RESIZE_SHAPE: (160, 160)
  AUGMENTATION:
    MIXUP_ALPHA: 0.2
MODEL:
  HEAD:
    NAME: simple_head_kaiming
    HIDDEN_DIMS: [500, 400, 300, 200]
    DROPOUT: 0.4
  SOLVER: 
    LOSS:
      NAME: label_smoothing_cross_entropy
```
In the experiment above, I decided to try a larger image `resize_shape` output, lowered my Mixup Alpha hyperparameter, tried a larger number of hidden dims, a different weight initialization in the model's head, added label smoothing, etc. All of which were not in the initial configs.  

### Step 4: Merge default configs with experiment configs and .env variables

We need a routine that overwrites default configs with the following priority: .env > YAML config > default config.

We first create a function inside config.py that will make a clone of the default configuration and can be called when training:
 ```python
def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()
```

Then we import this function in train.py and use `merge_from_file()` method to merge default configs with our YAML and .env files.
 ```python
def combine_cfgs(cfg_path):
    # Priority 3: get default configs
    cfg_base = get_cfg_defaults()    

    # Priority 2: merge from yaml config
    If path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(cfg_path)

    # Priority 1: merge from .env
    load_dotenv(find_dotenv(), verbose=True) # Load .env

    # Load variables
    path_overwrite_keys = ['DATASET.PATH_DATA_RAW',
                          os.getenv('DATASET.PATH_DATA_RAW'), 
                          'GCP_KEY',
                          os.getenv('GCP_KEY')]

    if path_overwrite_keys is not []:
        cfg_base.merge_from_list(path_overwrite_keys)

    return cfg_base
```

This function will combine all configs from all 3 files. We can now train our model with respect to our new configuration and be 100% sure that all desired settings are used.

### Step 5: Save the state and results of your model. Create backups

A good experimental framework should store all the results and configurations that are specific to an experiment. Therefore, we save the configuration settings at the start of our training module, then store the results and model stats after each epoch.

Create appropriate directories at the start of training: 

```python
    # Make output dir and its parents if they do not exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # Make backup folders if they do not exist
    backup_dir = os.path.join(output_path, 'model_backups')
    if not os.path.exists(backup_dir):
        os.mkdir(backup_dir)
    
    # Make result folders if they do not exist 
    results_dir = os.path.join(output_path, 'results')
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
```

Save the config file to `results` folder:

```python
config.dump(stream=open(os.path.join(self.results_dir, f'config{name_timestamp}.yaml'), 'w'))
```


Save the model state, optimizer state, scheduler state in `output_path` and create a backup of the model in `backup_dir`:

```python
# Create save_state dict with all hyperparamater + parameters
save_state = {
    "epoch": epoch + 1,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
              }

# If scheduler, add it to the save dict
if scheduler is not None:
    save_state['scheduler_state'] = scheduler.state_dict()
# Save model
torch.save(save_state, state_fpath)

# Save a backup
print("Making a backup (step %d)" % epoch)
backup_fpath = os.path.join(backup_dir, "model_bak_%06d.pt" % (epoch,))
torch.save(save_state, backup_fpath)
```
Save different performance metrics in `results_dir`:
```python
# Dump the traces
perf_trace.append(
    {
        'epoch': epoch,
        'train_err': train_total_err,
        'train_acc': train_total_acc,
        'train_kaggle_score': train_kaggle_score,
        'val_err': val_total_err,
        'val_acc': val_total_acc,
        'val_kaggle_score': val_kaggle_score,
        'lr': lr
    }
)
pickle.dump(perf_trace, open(results_dir, 'wb'))
```
We can store whatever metric we find relevant and wish to look at later. These metrics can also be sent to Tensorboard as we do with our project. I didn't mention it here, but we also need to be able to resume the training of a model by loading the saved state of our last checkpoint. Refer to the gitlab code for more details.  

### Step 6: Add CLI commands to specify an output_path and a cfg_path

In the final step, we add CLI functionality to be able to specify where we want everything to be saved while training with `output_path`, and from what config file we want to load configurations from: with `cfg_path`. Here is the code that does this in train.py:

```python
if __name__ == '__main__':
    # Docopt for command line arguments
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    output_path = arguments['-o']
    cfg_path = arguments['--cfg']
    cfg = combine_cfgs(cfg_path)
    cfg.OUTPUT_PATH = output_path
    train(cfg)
```
With this in place, we can use the following command to train our model and to run our experiment:

```
python -m src.tools.train -o experiments/exp10 --cfg src/config/experiments/exp10.yaml
```

That’s it! Now we can easily create multiple experiments, run them, and have everything from that experiement saved for later analysis. 

## Customize our Model's Architecture Based on our Configuration

The following section requires some familiarity with first-class functions, function decorators and closures. 
To repeat the context, our goal is to be able to pass in certain hyperparameters, settings and modules to a configuration file and have the code assemble the model correctly given these configs when we run our training routine.  

Consider the following config: 

```yaml
MODEL:
  HEAD:
    NAME: simple_head_module
    ACTIVATION: leaky_relu
  BACKBONE: se_resnext50
  SOLVER: 
    LOSS:
      NAME: label_smoothing_cross_entropy
```

Our training function will construct a model based on what was passed above. To illustrate, and assuming it was implemented, if we wanted to use a DenseNet121 instead we'd simply replace `se_resnext50` by `densenet121`. 

This is achieved thanks to a custom Registry class and function decorators. 

In short, the Registry class extends the dictionary data structure and provides an extra register function which will be used as a decorator to register our modules. 

(1). Registry class: creates a dictionary with a `register` function.

```python
HEAD_REGISTRY=Registry()
```
`HEAD_REGISTRY` is just an empty dictionary for now with an extra `.register()` method available to it. This is because the `Registry` class inherits from `dict`. 

```python
# Inherits from dict
class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides register functions.
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_module"]
    '''
    
    # Instanciated objects will be empyty dictionaries
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    # Decorator factory. Here self is a Registry dict
    def register(self, module_name, module=None):
        
        # Inner function used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # Inner function used as decorator -> takes a function as argument
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn # decorator factory returns a decorator function
```

(2) `register` method: decorator factory function that returns a `register_fn` decorator 

Given a `module_name` argument as well as an optional `module` argument, it returns a decorator function.

(3) `register_fn` decorator: gives access to a module function from a Registry dictionary

`register_fn` runs `_register_generic` which looks like the following: 

```python
def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module
```

This is what our decorator actually does: it takes a `module_dict`, `module_name` and `module` (which has to be a function), and organizes them in a dictionary-like structure. Access to the module is just like using a dictionary. 

To recap our decorator function:

```
register_fn input : a module function
register_fn output: some_registry["module_name"] = module_function
```

Here is an example of how all this is used in our framework: 

```python
# Instantiate Registry class
HEAD_REGISTRY=Registry()

# Call register function - see point (2) - which takes a module name as parameter 
@HEAD_REGISTRY.register('simple_head_module') 
def build_simple_pred_head(head_cfg):
    return SimplePredictionHead(head_cfg)
```

Function decorators are executed at import time. This means that when the code loads, Python will execute the `HEAD_REGISTRY.register('simple_head_module')` method (2). This will trigger our decorator to run and  the function `build_simple_pred_head` will be passed to `register_fn(fn)`. As a result, `HEAD_REGISTRY` dictionary will now have a new key assigned: `simple_head_module`, with a new value: `build_simple_pred_head` function - as per point (3).  

```python
HEAD_REGISTRY = {'simple_head_module': <function __main__.build_simple_pred_head(head_cfg)>}
```
This is great because it means that we now have an easy way to access this function! It is stored in our `HEAD_REGISTRY`, and can be accessed with `HEAD_REGISTRY['module_name']`. 

What does `build_simple_pred_head` function do exactly? It simply returns a `SimplePredictionHead` class which is responsible for creating the head, or classifier of our model. I won’t dive into the details of its code, but it's a normal Pytorch nn.Module class with multiple linear layers, activations, batch normalizations and a `forward` function. 

The last part we need to do is to run this function in order to instantiate the `SimplePredictionHead` class. In our framework, this is done with another `build` function. 

```python
def build_head(head_cfg):
    head_module = HEAD_REGISTRY[head_cfg['NAME']](head_cfg)
    return head_module
```
Calling `build_head` will then call the function located in our `HEAD_REGISTRY`, and return the instantiated `SimplePredictionHead` class. Here is the output:

```
SimplePredictionHead(
  (fc_layers): Sequential(
    (0): LinearLayer(
      (linear): Linear(in_features=1280, out_features=512, bias=True)
      (bn): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): LinearLayer(
      (linear): Linear(in_features=512, out_features=256, bias=True)
      (bn): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (2): LinearLayer(
      (linear): Linear(in_features=256, out_features=186, bias=True)
    )
  )
)
```

And there we have it. Specifying 'simple_head_module' in our configuration will trigger all the steps required to call `SimplePredictionHead`'s constructor and initialize the class, which can then be combine with the model's backbone somewhere else.  

This means that we can create multiple heads - all with different architectures, hyperparameters or weight initializations - and includ them in our model just by referencing their name in the config file. This is true for anything we'd like to be able to customize in our model: the backbone architecture, different loss functions, different schedulers. 

## Conclusion

I hope that I managed to illustrate the power of this highly customizable and flexible way to build the architecture of our model. Combined with a proper experimental framework as demonstrated above, and a neat and tidy project structure, we now have a solid framework and pipeline that can be used for any deep learning project. 

