
import argparse
import logging
import os

from src.extract import extract_information_blobs

log = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Extract blobs of information "
                    "from an image of a document.")

    parser.add_argument("document", type=str,
                        help="image file of a target document")
    parser.add_argument("--visualize", dest="visualize", action="store_true",
                        default=False,
                        help="show visualization of information blobs")
    parser.add_argument("--debug", dest="debug", action="store_true",
                        default=False,
                        help="set logging to debug level")
    parser.add_argument("--output", dest="output", type=str, default=None,
                        help="if provided, this program stores individual "
                             "information blobs extracted from the document "
                             "into the provided directory")
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(levelname).1s - %(message)s',
                        level=logging.DEBUG if args.debug else logging.INFO)

    if not os.path.isfile(args.document):
        log.error(f"The file {args.document} does not exist")
        exit(1)

    if args.output is not None and not os.path.isdir(args.output):
        log.error(f"{args.output} is not an existing directory")
        exit(2)

    boxes = extract_information_blobs(args.document,
                                      args.visualize,
                                      args.output)
    log.info(boxes)






