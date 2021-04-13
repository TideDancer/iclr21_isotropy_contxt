# Isotropy in the Contextual Embedding Space: Clusters and Manifolds

This repo contains the code, the poster and the slides for the paper: X.Cai, J.Huang, Y.Bian and K. Church, "Isotropy in the Contextual Embedding Space: Clusters and Manifolds", ICLR 2021.

The poster and slides are in the poster/ folder.

---

# Supplementary Code

## Set up using venv
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## [Optional]: install FAISS to compute Local Intrinsic Dimension (LID)
To install FAISS, please refer to https://github.com/facebookresearch/faiss

## Minimum effort to run the code
This script will generate a 3D plot of embeddings for tokens ["the","first","man"]

```bash
bash run_example.sh
```

---
## Generate embeddings:
For example, to generate BERT's 3rd layer's token embeddings, using wiki2 dataset, simply run the following:
```bash
source venv/bin/activate
python gen_embeds.py bert wiki2 3 --save_file bert.layer.3.dict
```
The code will put generated files at 
```
./embeds/[dataset]/[model].layer.[layerID].dict
```
The arguments include:
```bash
usage: gen_embeds.py model dataset layer

positional arguments:
  model                 model: gpt, gpt2, bert, dist (distilbert), xlm
  dataset               dataset: wiki2, ptb, wiki103, or other customized datapath
  layer                 layer id

optional arguments:
  -h, --help            show this help message and exit
  --save_file SAVE_FILE
                        save pickle file name
  --log_file LOG_FILE   log file name
  --datapath DATAPATH   customized datapath
  --batch_size BATCH_SIZE
                        batch size, default=1
  --bptt_len BPTT_LEN   tokens length, default=512
  --sample SAMPLE       [beta], uniform with probability=beta
  --no_cuda             disable gpu
```

---
## Perform comprehensive analysis
After obtain the embedding dict files in the previous step, we can perform comprehensive analysis by giving tasks arguments:
```bash
python gen_embeds.py embeds/wiki2/bert.layer.3.dict [tasks]
```

### Tasks include the following:

1. Compute the averaged inter-cosine similarity. For each type/word, sample 1 embedding instance.
```bash
--inter_cos --maxl 1
```

2. Compute the averaged intra-cosine similarity. 
```bash
--intra_cos
```

3. Perform clustering and report the mean-shifted, as well as clustered, inter and intra cosines.
```bash
--cluster --cluster_cos --center
```

4. Draw 2D, 3D or frequency heatmap figures.
```bash
--draw [2d]/[3d]/[freq]
```

5. Draw tokens in 3D plots. Specify tokens to draw using ``--draw_token``. The code will evaluate the string as a list, so please use the following format.
```bash
--draw token --draw_token "['the','first','man','&']"
```

6. Compute LID using either Euclidean distance or cosine distancel
```bash
--lid --lid_metric [l2]/[cos]
```

### Other settings include:
1. Dimension reduction and sampling. Please refer to -h to see the details of the following.
```bash
--embed [embed_dimension] --maxl [sample_method]
```

2. Center shiftting, subtract the mean.
```bash
--center
```

3. Zoom into the two distinct clusters existed in the GPT2 embedding sapce.
```bash
--zoomgpt2 [left]/[right] --draw 3d
```

---
## Get ELMo embeddings
We use AllenNLP package for ELMo. To obtain the embeddings, first need to prepare the input data as raw text file, then run:
```bash
source venv/bin/activate
python -m pip install allennlp==1.0.0rc3
allennlp elmo /path/to/dataset_text.txt /tmp/output.hdf5 --all
python elmo.py /tmp/output.hdf5 /path/to/dataset_text.txt [dataset_name] [layer_id]
```
Note that only allennlp latest version does not have "elmo" subcommand. So please use the older version, e.g 1.0.0rc3.
