# Perceiver CPI (Version 1.0)
A Pytorch Implementation of paper:

**PerceiverCPI: A nested cross-attention network for compound-protein interaction prediction**

Ngoc-Quang Nguyen , Gwanghoon Jang , Hajung Kim and Jaewoo Kang

Our reposistory uses https://github.com/chemprop/chemprop as a backbone for compound information extraction.
We highly recommend researchers read the paper [D-MPNN](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237) to better understand how it was used. 

Motivation: Compound-protein interaction (CPI) plays an essential role in drug discovery and is
performed via expensive molecular docking simulations. Many artificial intelligence-based approaches
have been proposed in this regard. Recently, two types of models have accomplished promising results in
exploiting molecular information: graph convolutional neural networks that construct a learned molecular
representation from a graph structure (atoms and bonds), and neural networks that can be applied to
compute on descriptors or fingerprints of molecules. However, the superiority of one method over the
other is yet to be determined. Modern studies have endeavored to aggregate information that is extracted
from compounds and proteins to form the CPI task. Nonetheless, these approaches have used a simple
concatenation to combine them, which cannot fully capture the interaction between such information.

Results: We propose the Perceiver CPI network, which adopts a cross-attention mechanism to improve
the learning ability of the representation of drug and target interactions and exploits the rich information
obtained from extended-connectivity fingerprints to improve the performance. We evaluated Perceiver CPI
on three main datasets, Davis, KIBA, and Metz, to compare the performance of our proposed model with
that of state-of-the-art methods. The proposed method achieved satisfactory performance and exhibited
significant improvements over previous approaches in all experiments

# 0.**Overview of Perceiver CPI**

![image](https://user-images.githubusercontent.com/32150689/169429361-cee1031f-fef3-43a6-9220-943fa21de233.png)


Set up the environment:

In our experiment we use, Python 3.9 with PyTorch 1.7.1 + CUDA 10.1.

```bash
git clone https://github.com/dmis-lab/PerceiverCPI.git
conda env create -f environment.yml
```

# 1.**Dataset and supplementary experiments**
![benchmark_data_vis](https://user-images.githubusercontent.com/32150689/167998111-f73c2fee-3ea4-49d4-8f60-8338e0acca00.PNG)


![image](https://user-images.githubusercontent.com/32150689/163341766-3115ffa6-0cfe-437e-be75-670de1b4da43.png)

The data should be in the format csv: 'smiles','sequences','label'!

The supplementary can be found: [HERE](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/39/1/10.1093_bioinformatics_btac731/2/btac731_supplementary_data.pdf?Expires=1683822877&Signature=h88F85eD2c911vsPpIRAlYyI-mrwjkgQoTGSPiY6KGCx5O6yrNZbU2UgDzuhBeYVx5RvLb-b1363ZDrebNYzh6pAnZ0Mq-h1li0aiXIVMJzeBD0~xLz9kzaR0DA09s2A7omblzmR690oeaMMvUjRiOOvbFYMmqmodcYZWxj7gGsIbwmamjQq~HERqDWEE5pBUeenht05ItvyKXZ~D2H1CLs2vbRaDAHMQl~vK9NljEan~pFPGJkofTdDPVOn34yszUlF7l231Omzge0T7CIiCC4dijeg1FfMrj-grJ2pqz1YC7Tcl1z22UvMd1pTcSXN1RveLZKMX~dI7GEJSeL2zg__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)


# 2.**To train the model:**
```bash
python train.py --data_path "datasetpath" --separate_val_path "validationpath" --separate_test_path "testpath" --metric mse --dataset_type regression --save_dir "checkpointpath" --target_columns label
```
_Usage Example:_
~~~
python train.py --data_path ./toy_dataset/novel_pair_0_train.csv --separate_val_path ./toy_dataset/novel_pair_0_val.csv --separate_test_path ./toy_dataset/novel_pair_0_test.csv --metric mse --dataset_type regression --save_dir regression_150_newprot_pre --target_columns label --epochs 150 --ensemble_size 3 --num_folds 1 --batch_size 50 --aggregation mean --dropout 0.1 --save_preds
~~~
# 3.**To take the inferrence:**
```bash
python predict.py --test_path "testdatapath" --checkpoint_dir "checkpointpath" --preds_path "predictionpath.csv"
```
_Usage Example:_
~~~
python predict.py --test_path ./toy_dataset/novel_pair_0_test.csv --checkpoint_dir regression_150_newprot_pre --preds_path newnew_fold0.csv
~~~
# 4.**To train YOUR model:**

Your data should be in the format csv, and the column names are: 'smiles','sequences','label'.

You can freely tune the hyperparameter for your best performance (but highly recommend using the Bayesian optimization package).

# 5.**Citations**
If you find the models useful in your research, please consider citing the relevant paper:

~~~
@article{10.1093/bioinformatics/btac731,
    author = {Nguyen, Ngoc-Quang and Jang, Gwanghoon and Kim, Hajung and Kang, Jaewoo},
    title = "{Perceiver CPI: A nested cross-attention network for compound-protein interaction prediction}",
    journal = {Bioinformatics},
    year = {2022},
    month = {11},
    abstract = "{Compound-protein interaction (CPI) plays an essential role in drug discovery and is performed via expensive molecular docking simulations. Many artificial intelligence-based approaches have been proposed in this regard. Recently, two types of models have accomplished promising results in exploiting molecular information: graph convolutional neural networks that construct a learned molecular representation from a graph structure (atoms and bonds), and neural networks that can be applied to compute on descriptors or fingerprints of molecules. However, the superiority of one method over the other is yet to be determined. Modern studies have endeavored to aggregate information that is extracted from compounds and proteins to form the CPI task. Nonetheless, these approaches have used a simple concatenation to combine them, which cannot fully capture the interaction between such information.We propose the Perceiver CPI network, which adopts a cross-attention mechanism to improve the learning ability of the representation of drug and target interactions and exploits the rich information obtained from extended-connectivity fingerprints to improve the performance. We evaluated Perceiver CPI on three main datasets, Davis, KIBA, and Metz, to compare the performance of our proposed model with that of state-of-the-art methods. The proposed method achieved satisfactory performance and exhibited significant improvements over previous approaches in all experiments.Perceiver CPI is available at https://github.com/dmis-lab/PerceiverCPISupplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac731},
    url = {https://doi.org/10.1093/bioinformatics/btac731},
    note = {btac731},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btac731/47214739/btac731.pdf},
}
~~~
