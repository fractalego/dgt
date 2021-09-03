Code for the paper [Semantic Reasoning with Differentiable Graph Transformations](https://arxiv.org/abs/2107.09579)
===
This repository contains the code that runs the toy examples in the paper.
While still a work in progress, the code is functional and illustrates the inner workings of the algorithm described in the article.

Installation
------------
The main requirements are installed with:

```bash
virtualenv --python=/usr/bin/python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

Download the glove vector files

```bash
cd data
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
echo "2196017 300" | cat - glove.840B.300d.txt > glove.txt
cd ..
```

Run the two test files
-----------------------

The system has been tested against two simple examples.
They can be run using the following commands

```bash
$ python -m dgt.train data/politics.json
```

```bash
$ python -m dgt.train data/more_gradient_rules.json
```
