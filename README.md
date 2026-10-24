# Sprachtechnologie1
Speechprocessing semester project in LMU-Phonetic
# The Final Assignment: Phone level Classification

This project builds a phone classifier using one simple Multi-Layer Perceptron(MLP). It trains the simple neural classifiers on Mel and MFCC acoustic features, based on the PD1 speech corpus. The main goal is to examine how the model performance changes, and we have two small sections to explain the model performance:

1. How performance changes with training data size
2. How performance changes with different feature type
3. How well the model transfers and generalizes across different recording sites

## Project Overview

The pipeline has four main stages:

1. Match each `.wav` audio file with its corresponding `.par` phone-alignment file.
2. Extract frame-level acoustic features from each utterance.
3. Convert phone labels into numeric classes and create train/validation/test splits.
4. Train MLP classifiers and evaluate them by changing the training size, feature type, and recording site.

Each dataset row corresponds to one acoustic frame. It contains the feature vector, frame index, utterance ID, speaker metadata, recording site, original phone label, and encoded numeric label.

## Basic Preparation Workflow and Reproducing the Pipeline

Run the notebooks in this order:

### Dataset reproducing
1. `build_hf_dataset.ipynb`
scans the corpus, pairs `.wav` and `.par` files by utterance ID, reads the MAU phone alignments, extracts acoustic features, and saves the resulting frame-level dataset as JSONL.

2. `random_shuffle.ipynb`
The dataset is split by speaker in `random_shuffle.ipynb`, so the same speaker does not appear in train, validation, and test at the same time. The shuffle means we shuffle all the speakers before assign them into different splits and also before the training dataset re-split, that makes sure that other features don't affect the spit result. 

3. `data_encoding.ipynb`
Phone labels are encoded and stored in `data_encoding.ipynb` using Hugging Face.

### How performance changes with training data size and different feature type
1. `model_for_mel/training_and_evaluation.ipynb` or `model_for_mfcc/training_and_evaluation.ipynb`

Run the two evaluation notebooks, train an MLP classifier on different training-set sizes:

```
train_25
train_50
train_75
train_100
```

Validation and test sets are kept fixed and unshuffled. This allows direct comparison of how training data size affects model performance. After running the model on all training splits, run the `compare_different_train_size` on both mel and mfcc folder. It compares the model performance across different training size.

2. To know how performance changes with different feature type, we can generally compare the model performance recorded in the evaluation folder.

### How well the model transfers and generalizes across different recording sites
1. We can reconstruct the dataset by running the `datas_by_site.ipynb`file, the entire dataset will then be organized by different recording site. The site with the smallest site-split cell is removed from the experiment. In the current setup, the kept sites are:

```
A = Kiel
N = Bonn
D = Munich
```

2. `model_for_mel_by_sites/training_and_evaluation.ipynb`
Run the training and evaluation notebooks, train the MLP classifier on different train-set-by-site:

```
training and validation (on A or D or N)
```
Basically, the notebook does the training, validation and testing as follows:

- train_site = A -> validation_site = A -> test_site = A/N/D

- train_site = N -> validation_site = N -> test_site = A/N/D

- train_site = D -> validation_site = D -> test_site = A/N/D


__When switching between Mel and MFCC experiments, different training scales or different dataset-by-sites, rerun the notebook from the dataset loading cell so that feature dimensions, label metadata, and model paths are updated consistently.__

## Repository Structure

```
.
├── build_hf_dataset.ipynb              # Build Mel/MFCC datasets from WAV + PAR files
├── random_shuffle.ipynb                # Speaker-based train/validation/test splitting
├── data_encoding.ipynb                 # Encode phone labels as ClassLabel IDs
├── datas_by_site.ipynb                 # Create recording-site-specific dataset splits(based on the encoded dataset)
├── dataset/
│   ├── mel/                            # Mel JSONL, split datasets, and encoded datasets
│   ├── mfcc/                           # MFCC JSONL, split datasets and encoded datasets
│   └── mel_by_site/                    # Mel datasets split by recording site
├── model_for_mel/
│   └── training_and_evaluation.ipynb   # Model for Mel examine by training-set size
├── model_for_mfcc/
│   └── training_and_evaluation.ipynb   # MFCC training by training-set size
├── model_for_mel_by_sites/
|    └── training_and_evaluation.ipynb   # Mel site-transfer experiments
├── rq1_different_train_size            # Analyze for the 
|   

```

## Model and Evaluation Outputs' Form

Each training run creates a timestamped folder under the corresponding model directory, for example:

```
model_for_mel/models/mel_train_100_full_eval_...
model_for_mel_by_sites/models/mel_train_site_A_...
```

Typical outputs include:

```
best_vloss.pt                    # Best model checkpoint by validation loss
results.json                     # Main metrics and training history
confusion_matrix.pt              # Confusion matrix for current experiments
evaluation/                     # Plots and CSV analysis files
```

In the evaluation folder, files are included as follows:

```
confusion_matrix_raw.png        # Plots for current confusion matrix, with raw counts
confusion_matrix_normalized.png # Plots for current confusion matrix, normalized to sumup 1

directional_confusions.csv      # CSV to explain which phone-pairs are more difficult to recognize, with directions, in which phone A -> phone B is different from phone B -> phone A
phone_pair_confusions.csv       # CSV to explain which phone-pairs are more difficult to recognize, without directions, in which phone A -> phone B is the same as phone B -> phone A

per_phone_results.csv           # CSV to record model's performance on each phones, including recall and precision rate

pause_data.json                 # Show the test accuracy without phone <p:> for pause
test_accuracy_comparison.png    # Plot comparing test accuracy with and without <p:>

training_validation_loss.png    # Plot comparing training and validation loss, including the information of on which epoch the model performance the best.
```

## Notes

- The split is speaker-based before site-specific filtering, which avoids speaker overlap between train, validation, and test.
- Site-transfer results are not fully controlled for training-set size, so differences may reflect recording-site effects, but affected slightly by the training size as well.
