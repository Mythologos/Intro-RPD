from argparse import ArgumentParser, BooleanOptionalAction
from os import listdir, mkdir, path

from aenum import NamedTuple
from natsort import natsorted
from tqdm import tqdm

from utils.cli.messages import BratMessage
from utils.data.converters.constants import BratAnnotation, FlagDict
from utils.data.converters.annotator import process_annotation_file
from utils.data.converters.xml_handler import produce_xml


BratFileTriple: NamedTuple = NamedTuple("BratFileTriple", ['text_file', 'annotation_file', 'output_file'])


def process_brat_data(text_filepath: str, annotation_filepath: str, output_filepath: str, flags: FlagDict):
    # First, we get the annotations and their locations.
    annotation_queue: list[BratAnnotation] = process_annotation_file(annotation_filepath)
    should_process_file: bool = False if (flags["truncation"] is True and len(annotation_queue) == 0) else True
    if should_process_file is True:
        produce_xml(text_filepath, annotation_queue, output_filepath, flags)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument("input_filepath", type=str, help=BratMessage.INPUT_FILEPATH)
    parser.add_argument("output_filepath", type=str, help=BratMessage.OUTPUT_FILEPATH)
    parser.add_argument("--capitalization", action=BooleanOptionalAction, default=True, help=BratMessage.CAPITALIZATION)
    parser.add_argument("--punctuation", action=BooleanOptionalAction, default=True, help=BratMessage.PUNCTUATION)
    parser.add_argument(
        "--punctuation-strategy", type=str, default="preserve", choices=["preserve", "exclude"],
        help=BratMessage.STRATEGY
    )
    parser.add_argument("--sectioning", action=BooleanOptionalAction, default=True, help=BratMessage.SECTIONING)
    parser.add_argument("--truncation", action=BooleanOptionalAction, default=True, help=BratMessage.TRUNCATION)
    args = parser.parse_args()

    # First, we handle the output filepath. If it doesn't exist, we create it.
    if not path.exists(args.output_filepath) and path.exists(args.output_filepath.rsplit(".", maxsplit=1)[0]):
        mkdir(args.output_filepath)
    elif not path.exists(args.output_filepath):
        raise ValueError("Inappropriate output directory provided. Please try again.")

    # First, we check whether the input and output directory exist.
    # The output directory may not exist, in which case it is created.
    if not path.exists(args.input_filepath) or not path.isdir(args.input_filepath):
        raise ValueError("Invalid filepath given for input directory. Please try again.")
    else:
        input_files: list[str] = listdir(args.input_filepath)
        input_files = natsorted(input_files)
        file_triples: list[BratFileTriple] = []
        input_file_index: int = 0
        while input_file_index < len(input_files):
            annotation_filename: str = input_files[input_file_index]
            text_filename: str = input_files[input_file_index + 1]

            # We check to make sure that the filenames are the same as a sanity check,
            # and we assure that extensions are correct.
            annotation_filename_title, annotation_filename_extension = annotation_filename.rsplit('.', maxsplit=1)
            text_filename_title, text_filename_extension = text_filename.rsplit('.', maxsplit=1)
            if annotation_filename_title != text_filename_title:
                raise ValueError(f"File <{annotation_filename}> does not match with <{text_filename}>. "
                                 f"Please appropriately name and order files.")
            elif annotation_filename_extension != "ann":
                raise ValueError(f"File <{annotation_filename}> does not have the appropriate .ann extension. "
                                 f"Please relabel it or reorder the files and try again.")
            elif text_filename_extension != "txt":
                raise ValueError(f"File <{text_filename}> does not have the appropriate .txt extension. "
                                 f"Please relabel it or reorder the files and try again.")
            else:
                new_brat_triple: BratFileTriple = BratFileTriple(
                    f"{args.input_filepath}/{text_filename}",
                    f"{args.input_filepath}/{annotation_filename}",
                    f"{args.output_filepath}/{text_filename_title}_annotated.xml"
                )
                file_triples.append(new_brat_triple)

            input_file_index += 2

    format_flags: FlagDict = {
        "capitalization": args.capitalization,
        "punctuation": args.punctuation,
        "punctuation_strategy": args.punctuation_strategy,
        "sectioning": args.sectioning,
        "truncation": args.truncation
    }

    for (input_path, annotation_path, output_path) in tqdm(file_triples):
        process_brat_data(input_path, annotation_path, output_path, format_flags)
