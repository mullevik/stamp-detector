# stamp-detector
Detect stamps in scanned documents (PoC).

## Unsupervised clustering
Uses kmeans to extract individual blobs of the document in hopes of extracting
stamps as a single blob.

## Supervised clustering
For each extracted blob, determine, if the blob contains a stamp.
(Not implemented yet)
