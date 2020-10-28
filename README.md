# ATJ-Net
An Auto-Table-Join Network for automatic learning on relational databases. This repository contains a part of experiments results on Yelp dataset and KddCup dataset.

## Download the dataset
Download the Yelp dataset from https://www.yelp.com/dataset to the directory `./raw/kddcup` and the Kddcup dataset from https://www.4paradigm.com/competition/kddcup2019 to the directory `./raw/yelp`.

After downloading and decompressing, please arrange the file as follows:
```
raw
├── kddcup
│   ├── A
│   │   ├── test
│   │   │   └── main_test.data
│   │   └── train
│   │       ├── info.json
│   │       ├── main_train.data
│   │       ├── main_train.solution
│   │       ├── table_1.data
│   │       ├── table_2.data
│   │       └── table_3.data
...
...
└── yelp
    ├── business.json
    ├── checkin.json
    ├── Dataset_Challenge_Dataset_Agreement.pdf
    ├── photo.json
    ├── review.json
    ├── tip.json
    ├── user.json
    └── Yelp_Dataset_Challenge_Round_13.pdf
```

## Preprocess the dataset to tabular format

Transfer the raw dataset to tabular format and build the sub-dataset.
```
cd ./preprocess
python kddcup_preprocess.py
python yelp_preprocess.py
python yelp_build_subdataset.py
```

After preprocessing, the files should be as follows:
```
data
├── a
│   ├── info.json
│   ├── main.data
│   ├── table_1.data
│   ├── table_2.data
│   └── table_3.data
...
...
y1
├── business.data
├── info.json
├── review.data
├── review_text.emb
├── tip.data
├── tip_text.emb
└── user.data
...
...
```
The directories `a-e` are the sub-datasets from Kddcup, and `y1-y5` are the sub-datasets from Yelp.

Each directory is a database, the `info.json` is the schema file of the database, and `*.data` are the table files.

## Compile the C++ Codes
We use C++ to accelerate the operator of Python and PyTorch, and the compiling command has been written in the makefile:
```
cd ./src
```

Note that, we need to replace Python lib location in the file `src/makefile` line two by:
```PYPATH = -L${Your_Python_Lib_location} -I${Your_Python_Include_location} -lpython3.7m```

Then
```make```

## Training
Running the ATJ-Net on second sub-dataset of Yelp and we shall see the results.
```
python train.py --database_path=../data/y2
```

The detailed command-line arguments are used as follows:
```
usage: train.py [-h] [--database_paths DATABASE_PATHS]
                [--preprocess PREPROCESS] [--mode MODE] [--cpu CPU]
                [--gpu GPU] [--keval KEVAL]

optional arguments:
  -h, --help            show this help message and exit
  --database_paths DATABASE_PATHS
                        Database paths
  --preprocess PREPROCESS
                        Preprocessing:
                        basic/simple/proper/full/complex
  --mode MODE           
                        Algorithm mode: 
                        atj0_default_param/atj1_default_param/atj2_default_param/
                        atj0_bayes_opt/atj1_bayes_opt/atj2_bayes_opt
                        atj1_random_bayes_opt/atj2_random_bayes_opt/lgbm_default_param/lgbm_bayes_opt
  --cpu CPU             Which gpu to use
  --gpu GPU             Which gpu to use
  --keval KEVAL         Cross evaluation number
```
