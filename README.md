<!-- <p align="center">
  <img src="WavRx_logo.png" alt="WavRx logo" width=200/>
</p> -->

# Audio Health Benchmark

This repository provides scripts for training and evaluating your customized model on **17 datasets covering 11 different pathologies**.

This repository can be used to (1) conduct training of *WavRx* on the 6 datasets; (2) run inference using the pretrained *WavRx* backbones; (3) train and test your self-customized models on the 6 datasets without efforts needed for editing training/evaluation scripts.

For detailed information, refer to [paper]():

```bibtex
PLACEHOLDER
```

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Dependencies](#-dependencies)
- [Pretrained model](#-pretrained-model)
- [Datasets and Recipes](#-Datasets-and-Recipes)
- [Quickstart](#-quickstart)
  - [Running a single task](#Running-a-single-task)
  - [Running multiple tasks](#Runnin-multiple-tasks)
- [Train and test your own model](#-Train-and-test-your-own-model)
- [Results](#-results)
- [Contact](#-contact)
- [Citing](#-citing)

# 🛠️ Dependencies

We use *PyTorch* and *SpeechBrain* as the main frameworks.


1. Clone the repository:
   ```shell
   git clone https://github.com/zhu00121/Audio-Health-Benchmark
   cd Audio-Health-Benchmark
   ```
2. Create a virtual env for the repo
   ```
   python3.10.13 -m venv <NAME_YOUR_VENV>
   source <NAME_YOUR_VENV>/bin/activate
   ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

# 🌟 Pretrained Model Backbones (to be released on HG)
Note that some employed datasets are subject to confidentiality agreement, this restriction may also apply to the pretrained model weights. We are currently working on making the pretrained backbones open-source on HuggingFace.
| **Model**                                                                 | **Dataset**                                                                                       | **Repo**                                                         |
|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| X                      | [Cambridge COVID-19 Sound]()                                                                                     | [huggingface.co/](https://huggingface.co/)  |
| Y                       | [DiCOVA2]()                                                                                     | [huggingface.co/](https://huggingface.co/)  |
| Z                    | [TORGO]()                                                                                     | [huggingface.co/](https://huggingface.co/)  |


# 📈 Results of Baseline Models

TBD

# 👷 Datasets

Majority of the datasets require agreements to be signed for obtaining access. Please refer to the **Download links** in the table below to go to the data download pages and follows their instructions to obtain the data. Once the data are downloaded, refer to the data prepration guide which helps to prepare the data in the required format.

| **Dataset**                              | **Task**                             | **Download links** | **Data preparation guide**                                                                       |
|------------------------------------------|--------------------------------------|----------------------------------------------------------------------------------------|--------------------------------------|
| Cambridge-Task1 | Respiraty Symptom Detection  | [TBD]()                                               |``exps/Cambridge_Respiratory/Guide.ipynb``|
| Cambridge-EN                         | Respiraty Symptom                   | [TBD]()|``exps/Cambridge_Respiratory_Task1/Guide.ipynb``|
| DiCOVA2                                | COVID-19  | [TBD]()| ``exps/DiCOVA2/Guide.ipynb`` |                  
| TORGO                                  | Dysarthria   | [Link](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)| ``exps/TORGO/Guide.ipynb`` |
| Nemours                          | Dysarthria | [TBD]()| ``exps/Nemours/Guide.ipynb`` |
| NCSC                                   | Cervical Cancer  | [TBD]()| ``exps/NCSC/Guide.ipynb`` |

Once the data are downloaded, they can be automatically prepared in the required format by running the following code:
```
PLACEHOLDER FOR CODE
```

## 

# ▶️ Quickstart

## Running a single task
Since each dataset has a different dataset structure with the corresponding partition, the training receipes are therefore stored separately in different folders. *Links* in the table above can be used to locate the corresponding recipes for a given dataset. 

The steps for model training are as follows:

1. Medical data are hard to access and they typically do not have similar data structures. We make this easy for you. Use the **Link to data preparation scripts** to see where to download the data files, and how to prepare each dataset in the required format.

2. [Optional only if you want to train your own model] Place the code of your model in the ``model`` folder, it needs to have 1 output neuron (without sigmoid).

3. Check the hyperparameter file at ``exps/<DATASET>/hparams/<DATASET>.yaml``. Modify the variables if needed. We provide a detailed guidance in ``demos/demo_hparam.md`` where we walk through the hyperparam file and demonstrate how to modify it for your own usage. If you simply want to replicate our results, there is no need to change it.

4. The ``train.py`` does NOT need to be edited. Unless you want to change the training strategy or the loss function (i.e., Supervised training with BCEwithlogits loss). All the hyperparameters and the input models are controlled by modifying the hyperparam file. This helps to ensure that models are compared in a fair manner.

5. Initiate training by calling ``python train.py hparams/<DATASET>.yaml``. The test evaluation will be automatically conducted at the end of training using the best checkpoint with the highest F1 score. The results will be automatically saved.

6. Repeat step-5 for each dataset, results will be saved independently in the corresponding `exp/<DATASET>` folder.

😎 Voila! Enjoy your model training 😎

## Running multiple tasks in one-shot
TBD




# 📧 Contact

For questions or inquiries, feel free to open an issue or you can reach us at ([yi.zhu@inrs.ca](mailto:yi.zhu@inrs.ca)).
<!-- ############################################################################################################### -->
# 📖 Citing

If you use *WavRx* and/or its backbones and/or tge training recipes, please cite:

```bibtex

```
