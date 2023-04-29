# Preparing NaQ datasets for augmentation

We provide two modes of creating the NaQ dataset.
1. **[RECOMMENDED]** Download pre-generated dataset used for our paper experiments using the ego4d api.
2. Create NaQ datsets from scratch.

See instructions for both options below.

## Option 1: Download pre-generated NaQ datasets

* We share pre-generated datasets through the [Ego4D api](https://ego4d-data.org/docs/CLI/).
    ```
    ego4d --output_directory="<OUTPUT DIRECTORY>" --datasets naq_datasets --aws_profile_name <AWS-PROFILE-WITH-EGO4D-ACCESS> --version v2
    ```
* Copy the data `$NAQ_ROOT` and extract.
    ```
    cp `<OUTPUT DIRECTORY>/v2/baselines/naq_datasets.zip` $NAQ_ROOT/
    unzip naq_datasets.zip
    ```

## Option 2: Creating NaQ datasets from scratch

* Download the Ego4D narrations in the EgoClip format provided [here](https://github.com/showlab/EgoVLP#egoclip-an-egocentric-video-language-pretraining-dataset) to `$NAQ_ROOT/data/egoclip.csv`.
* Convert EgoClip narrations to NaQ dataset.
    ```
    cd $NAQ_ROOT
    python utils/create_naq_dataset.py --type nlq
    python utils/create_naq_dataset.py --type tacos
    ```
* Prepare datasets for NLQ training.
    ```
    cd $NAQ_ROOT
    # Prepare NLQ dataset
    python utils/prepare_ego4d_dataset.py \
        --input_train_split data/nlq_train.json \
        --input_val_split data/nlq_val.json \
        --input_test_split data/nlq_test_unannotated.json \
        --output_save_path data/dataset/nlq_official_v1

    # Prepare NLQ + NaQ dataset
    python utils/prepare_ego4d_dataset.py \
        --input_train_split data/nlq_aug_naq_train.json \
        --input_val_split data/nlq_val.json \
        --input_test_split data/nlq_test_unannotated.json \
        --output_save_path data/dataset/nlq_aug_naq_official_v1

    # Prepare TaCOS dataset
    python utils/prepare_ego4d_dataset.py \
        --input_train_split data/tacos_train.json \
        --input_val_split data/tacos_val.json \
        --input_test_split data/tacos_val.json \
        --output_save_path data/dataset/tacos_official_v1

    # Prepare TaCOS + NaQ dataset
    python utils/prepare_ego4d_dataset.py \
        --input_train_split data/tacos_aug_naq_train.json \
        --input_val_split data/tacos_val.json \
        --input_test_split data/tacos_val.json \
        --output_save_path data/dataset/tacos_aug_naq_official_v1
    ```
