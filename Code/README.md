# Pipeline - TCA

This Pipeline is meant to analyze real-world calcium imaging data through Tensor Component Analysis (TCA).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Tensorly library is used to optimize TCA 

```
pip install tensorly==0.4.3
```

It is best if CUDA is configured on your system, it will help the optimization of the TCA go faster. PyTorch will distribute work to the GPUs.

````
pip install torch
````

### Installing

Clone the repository and create a Data, an Output and a Figure folder. 

```
git clone https://github.com/cguerino/pipeline_tca.git
```

### Setting up

In``Code/functions/settings.py`` ,modify paths entries specifying the path of your Data, Output and Figure folder created before. A fixed parafac folder path can be specified if you foresee using this method. 

## Examples

### Preprocessing

First, data needs to be pre-processed. This is done on an animal basis. Please launch ``Code/preprocessing.py`` and specify an animal (index of the animal name in settings' list). Additional arguments are available, see them with ``--help``.

```
preprocessing.py -a 14 -co 20
```

This will run preprocessing algorithm for animal 14 in the settings' list with a cutoff of consecutive NaNs to 20.

### TCA

Tensor Component analysis can now be applied to the preprocessed data. To do this, use ``Code/TCA.py``.

```
TCA.py -a 14 -E 3 -f non_negative_fixed_parafac -i random -v -fa 10 11 12 -fE 1 2 3
```

Here, we perform TCA analysis (refer to the Code for detailed steps) ontrials from experiment 3 of the animal 14. We use a non_negative_fixed_parafac with random initialization. Time factor is build on a daily basis with experiment 1, 2 and 3 from animal 10, 11 and 12.

### Meta_analysis

Some meta_analysis can be performed using ``meta_analysis.py`` or ``neural_network.py``. Arguments are passed the same way.

## Built With

* [Tensorly](https://github.com/tensorly/tensorly/) - A matrix decomposition library

## Authors

* **Antonin Verdier** - *Master student*
* **Corentin Guerinot** - *Ph.D. student*

