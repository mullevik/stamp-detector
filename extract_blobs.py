

import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract blobs of information "
                    "from an image of a document.")

    parser.add_argument("document", type=str,
                        help="image file of a target document")
    parser.add_argument("--visualize", dest="visualize", action="store_true",
                        default=False,
                        help="show visualization of information blobs")
    parser.add_argument("--output", dest="output", type=str, default=None,
                        help="if provided, this program stores individual "
                             "information blobs extracted from the document "
                             "into the provided directory")

    args = parser.parse_args()







