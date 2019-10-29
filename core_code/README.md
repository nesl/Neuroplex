# Core codes for Neuroplex
Neuroplex for on complex event detection

Codes are from the MNIST CE experiment, can be easily adjusted to other dataset. (changing input/ouput data, structure of perception model, parameter of integrated model...)

#### Files description:

###### 1 .py files
- __utils.py__:
Contains the core codes for data generation, annotation, train NRlogic model, create combined model (didn't use timeshift, but use customized layer in Keras)

- __neurallogic.py:__
contains the codes for training a NRlogic model

- __lenet.py__:
Codes for a general LeNet for MNIST recognition. The accuracy trained in standard way should be around 99.2%

- __data_prepare.py:__
Prepare for both "audio complex event data" and "customized audio event data" (used for testing perception model)
    - __Todo:__ add parameter in function input, to enable the "valid" option for generating only valid data. (contains CE.)
    - Codes for that is already used in the audio experiement.
- __intergrated_model.py__:
Integrates the NRlogic model and perception model, and compile it with appropriate loss (MSE loss + lambda x semantic loss). Use Adam 0.001 for training.

- __grid_search.py__:
Codes for searching the correct lambda (trade off parameter for MSE loss and semantic loss)

-__run_exp.py__:
Running the simulation for sim1 in the paper. Compare the proposed method with other baselines: ablation, scratch, c3d.


## Dimension of complexity:
* Length of event window
* Number of unique events 
* Number of events contained in Complex Events

Adding CE types is actually providing more annotation info, so it can somehow simply the problem.
