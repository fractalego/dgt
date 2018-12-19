DGT
===
Preliminary code


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

