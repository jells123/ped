import argparse
import logging

import download_data

LOGGING_FORMAT = "%(asctime)-15s"


def parse_args():
    parser = argparse.ArgumentParser(description="Youtube trending videos")
    parser.add_argument(
        "--drive_folder_id", type=str, nargs="?", const=True, default="",
        help="Downloads data from google drive folder_id if not specified, it doesn't download anything"
    )
    parser.add_argument(
        "--preprocess", type=str2bool, nargs="?", const=True, default=True, help="Activates preprocessing of data"
    )
    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    logging.basicConfig(format=LOGGING_FORMAT)
    logger = logging.getLogger("PED")
    args = parse_args()

    logger.info(f"input config: {args}")

    if args.drive_folder_id:
        download_data.download(args.drive_folder_id)

    # TODO missing values

    # TODO column transformation + additional features

    # TODO grouping by video_id ?

    # TODO predict


if __name__ == "__main__":
    main()
