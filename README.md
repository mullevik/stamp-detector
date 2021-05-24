# stamp-detector
Detect stamps in scanned documents (PoC).

## Install
Create a virtual environment 
```shell script
python3.8 -m venv env
source env/bin/activate
```
and install dependencies from ```requirements.txt```.
```shell script
pip install -r requirements.txt
```

## Unsupervised clustering
Uses kmeans to extract individual blobs of the document in hopes of extracting
stamps as a single blob.

Run the information blob extraction script like this:
```shell script
python extract_blobs.py data/documents/stamps/faktura-237279.pdf.jpg --visualize
```

## Supervised clustering
For each extracted blob, determine, if the blob contains a stamp.
(Not implemented yet)
