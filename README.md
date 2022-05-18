# PerceiverCPI (Version 1.0)
A Pytorch Implementation of paper:

**PerceiverCPI: An cascaded cross-attention network for compound-protein interaction prediction**

Ngoc-Quang Nguyen , Gwanghoon Jang , Hajung Kim and Jaewoo Kang

Our reposistory uses https://github.com/chemprop/chemprop as a backbone for compound information extracting

Motivation: Drug discovery using traditional methods is labor-intensive and time-consuming; hence,
efforts are being made to repurpose existing drugs. To find new methods for drug repurposing, many
artificial intelligence-based approaches have been proposed to predict compound-protein interactions
(CPIs). However, owing to the high-dimensional nature of datasets extracted from drugs and targets,
computational models for understanding CPIs play an important role in facilitating drug repurposing in
modern life. Recently, two types of models have achieved promising results for exploiting molecular
information: graph convolutional neural networks that construct a learned molecular representation from
the graph structure (atoms and edges) of the molecule and neural networks that can be applied to computed
molecular descriptors or fingerprints. However, the superiority of one method over the other is yet to
be determined. Numerous recent studies have attempted to minimize the limitations of the restricted
capacity of graph neural networks in small datasets, which include meaningless initial embeddings and
lack of expressivity. However, these approaches commonly use simple concatenation of separate networks
(compound net and protein net); therefore, there is no interaction between such information. To mitigate
these limitations, we propose the PerceiverCPI network, which adopts the cross-attention mechanism to
improve the learning ability of the representation of drug and target interactions and takes advantage of
the rich information obtained from extended-connectivity fingerprints (ECFP) to improve the performance.

Results: We perform PerceiverCPI on three main datasets, Davis, KIBA, and Metz, to compare our
proposed model with state-of-the-art methods. The proposed method offers satisfactory performance and
significant improvements over previous approaches in all experiments

![model_architecture](https://user-images.githubusercontent.com/32150689/160957444-42323b80-c516-45d3-a67c-c896b150c252.PNG)





Set up the environment:

```bash
git clone https://github.com/dmis-lab/PerceiverCPI.git
conda env create -f environment.yml
```

# 1.**Dataset**
![benchmark_data_vis](https://user-images.githubusercontent.com/32150689/167998111-f73c2fee-3ea4-49d4-8f60-8338e0acca00.PNG)


![image](https://user-images.githubusercontent.com/32150689/163341766-3115ffa6-0cfe-437e-be75-670de1b4da43.png)

The data should be in the format csv: 'smiles','sequence','label'!

The dataset can be found: [HERE](https://drive.google.com/drive/folders/1I7LWz4MwlR62dk__GNvIoyxN5sAUrzrf?usp=sharing)

The pretrained model can be found: [HERE](https://drive.google.com/drive/folders/16Qte7qn9Erq4jUCBcOC7WhaX7xImC2SZ?usp=sharing)

# 2.**To train the model:**
```bash
python train.py --data_path "datasetpath" --separate_val_path "validationpath" --separate_test_path "testpath" --metric mse --dataset_type regression --save_dir "checkpointpath" --target_columns label
```
_For example:_

python train.py --data_path /hdd1/quang_backups/dti/mpnn_2021/data_fold0/davis_newprot_0/0davis_1_train_newprot.csv --separate_val_path /hdd1/quang_backups/dti/mpnn_2021/data_fold0/davis_newprot_0/0davis_val_newprot.csv --separate_test_path /hdd1/quang_backups/dti/mpnn_2021/data_fold0/davis_newprot_0/0davis_test_newprot.csv --metric mse --dataset_type regression --save_dir regression_150_newprot_pre --target_columns label --epochs 150 --ensemble_size 3 --num_folds 1 --batch_size 50 --aggregation mean --dropout 0.1 --save_preds

# 3.**To take the inferrence:**
```bash
python predict.py --test_path "testdatapath" --checkpoint_dir "checkpointpath" --preds_path "predictionpath.csv"
```
_For example:_

python predict.py --test_path /hdd1/quang_backups/dti/mpnn_2021/data/deeppurpose/5_folds_check/davis/newnew/fold0/0davis_test_newprot.csv --checkpoint_dir regression_150_newprot_pre --preds_path newnew_fold0.csv
