from aenum import NamedConstant


class BIOMessage(NamedConstant):
    INPUT_FILEPATH: str = "existing input directory containing compatible XML parallelism data"
    MAPPINGS: str = "tag mappings to be used to combine discrete nodes into common classes for the AdjacencyList"
    OUTPUT_FILEPATH: str = "path to output file (sans extension) where the resulting " \
                           "AdjacencyList's statistics will be stored"
    SAVE_FORMATS: str = "formats in which information about the AdjacencyList data should be saved"
    SAVE_FREQUENCY: str = "style in which information about edge counts should be used and/or saved"
    SAVE_ORDER: str = "order in which heatmap nodes should be depicted"
    SAVE_PATH: str = "location where specially-formatted AdjacencyList data should be saved"


class BootstrapperMessage(NamedConstant):
    FIRST_FILEPATH: str = "location of first dataset directory for bootstrapping procedure"
    SECOND_FILEPATH: str = "location of second dataset directory for bootstrapping procedure"
    ALPHA: str = "value determining confidence interval"
    CLEANERS: str = "preprocessors to clean up parallelism data before matching"
    OUTPUT_FILE: str = "output filepath for bootstrapping results"
    SAMPLE_COUNT: str = "number of samples to perform for bootstrap estimate"
    SAMPLE_PERCENTAGE: str = "percentage of samples, relative to original number of samples, " \
                             "to extract per bootstrap sample"


class BratMessage(NamedConstant):
    INPUT_FILEPATH: str = "directory containing text and annotation files from brat"
    OUTPUT_FILEPATH: str = "directory to contain XML outputs built from brat's files"
    CAPITALIZATION: str = "flag determining whether the data will be capitalized or not"
    PUNCTUATION: str = "flag determining whether the data will contain punctuation or not"
    STRATEGY: str = "procedure for determining how punctuation will interact with parallelism tags;" \
                    "the 'preserve' strategy will keep punctuation as the .ann file labels it, " \
                    "whereas the 'exclude' strategy will bump punctuation out of the edges of any branch"
    SECTIONING: str = "flag determining whether section breaks will be included or not"
    TRUNCATION: str = "flag determining whether text files without annotation data " \
                      "will be represented in the final output or not"


class SplitMessage(NamedConstant):
    INPUT_DIRECTORY: str = "an existing directory containing XML parallelism data"
    OUTPUT_DIRECTORY: str = "an existing directory where specified data splits will be stored"
    MATCH_DIRECTORY: str = "an existing directory where XML parallelism data of the same kind has already been split"
    SETS: str = "names of the splits"
    RATIOS: str = "ratios (relative to 1) of the data per split"


class HyperparameterMessage(NamedConstant):
    EMBEDDING: str = "fixed model embedding type for all trials"
    ENCODER: str = "fixed model encoder type for all trials"
    VARIED: str = "hyperparameters which will be varied across all trials"
    SPECIFIED: str = "hyperparameter names and values which will be fixed across all trials"
    CONSTRAINTS: str = "constraints to apply to hyperparameter trial generation"
    TEST_PARTITION: str = "name of data partition used for evaluation scripts"
    TRIALS: str = "number of hyperparameter trials to generate"
    TRIAL_OFFSET: str = "starting number for hyperparameter trial generation"
    OUTPUT_FILENAME: str = "output filename prefix for trial files"
    OUTPUT_FORMAT: str = "output filetype for trial files"


class NeuralMessage(NamedConstant):
    ACTIVATION_FUNCTION: str = "activation function used in feedforward layers of a Transformer encoder"
    BLENDER: str = "blending function used in Encoder-CRF architecture to combine subword embeddings"
    DATASET: str = "name of dataset to be used"
    DATA_SPLITS: str = "names of splits available for the given dataset"
    DISPLAY_COUNT: str = "number of nonempty outputs to print or write per epoch"
    DROPOUT: str = "decimal representing the probability of dropout for neurons in the Transformer encoder"
    EMBEDDING_FILEPATH: str = "path to location of relevant word embeddings"
    EMBEDDING: str = "Encoder-CRF embedding type"
    ENCODER: str = "Encoder-CRF encoder type"
    EPOCHS: str = "maximum number of epochs for training; must be included if patience is not"
    EVALUATION_PARTITION: str = "name of data partition used for evaluating the model; " \
                                "it is the validation set during training and the test set during testing"
    HEADS: str = "number of attention heads per Transformer encoder layer"
    FROZEN_EMBEDDINGS: str = "a flag determining whether pretrained BERT-based embeddings " \
                             "will be subject to continued training or not"
    HIDDEN_SIZE: str = "size of the relevant encoder hidden state"
    INPUT_SIZE: str = "size of the relevant embedding state"
    LAYERS: str = "number of encoder layers"
    LEARNING_RATE: str = "value representing the learning rate used by an optimizer during training"
    LEMMATIZATION: str = "a flag determining whether input tokens are lemmatized " \
                         "(before subword tokenization, if applicable)"
    MODE: str = "indication of whether a model is being newly trained or both loaded and evaluated"
    OPTIMIZER: str = "optimization algorithm used during training"
    OUTPUT_DIRECTORY: str = "directory path to which actual tagged model evaluation results will be written"
    PATIENCE: str = "maximum number of epochs without validation set improvement before early stopping; " \
                    "must be included if epochs is not"
    PRETRAINED_FILEPATH: str = "path to location of relevant pretrained BERT model"
    PRINT_STYLE: str = "determines what statements are printed to the console during execution"
    REPLACEMENT_PROBABILITY: str = "probability that some token will be replaced with <UNK> during training " \
                                   "for models with word-level embeddings"
    REPLACEMENT_STRATEGY: str = "strategy by which tokens are deemed viable to be replaced with " \
                                "<UNK> during training for models with word-level embeddings"
    TOKENIZER_FILEPATH: str = "filepath to subword tokenizer, if applicable"
    TQDM: str = "flag for displaying tqdm-based iterations during processing"
    TRAINING_PARTITION: str = "named split of the data used for training"
    VISUALIZE: str = "names of visualization to generate on the basis of training performance"
    VISUALIZATION_DIRECTORY: str = "directory path where visualizations created will be stored"
    WEIGHT_DECAY: str = "numeric value for optimizer weight decay"


class GatherMessage(NamedConstant):
    DIRECTORIES: str = "list of directories to traverse for gathering result data"
    FILE_TYPE: str = "type of file to gather result data from"
    FILE_REGEX: str = "Python-style regular expression used for permitting the inclusion of files in result compilation"
    OUTPUT_FILE: str = "filepath to (possibly nonexistent) result compilation file"
    SUBDIRECTORY_REGEX: str = "Python-style regular expression used for excluding subdirectories in result compilation"


class ParallelismMessage(NamedConstant):
    INPUT_DIRECTORY: str = "existing directory path containing XML parallelism data"
    OUTPUT_DIRECTORY: str = "directory path to contain statistical information for input data"
    AGGREGATE: str = "flag to determine whether aggregate stats will be computed"
    VISUALIZE: str = "flag to determine whether matplotlib-based visualizations will be generated"


class PSEMessage(NamedConstant):
    ANNOTATION_DIRECTORY: str = "directory containing word-level brat parallelism annotations"
    OUTPUT_DIRECTORY: str = "directory to contain output PSE-I data"
    PSE_DIRECTORY: str = "directory containing original XML-style sentence-level parallelism data"


class AnalyzerMessage(NamedConstant):
    ALPHA: str = "offset to determine p-value (e.g., an alpha of 0.95 generates a p-value of 0.05)"
    ANALYSIS_TYPE: str = "type of analysis to perform over results data"
    CRITERIA: str = "selection of filter criteria for data"
    INPUT_FILE: str = "CSV input filepath containing results information"
    OUTPUT_FILEPATH: str = "filepath used to store outputs; currently only used for boxplot creation"
    SPLIT: str = "split of data examined for results analysis"


class GenericMessage(NamedConstant):
    COLLECTION_FORMAT: str = "designation of textual units in parallelism data"
    MODEL_LOCATION: str = "directory in which a model is or will be contained"
    MODEL_NAME: str = "filename prefix for a model at a designated location (see model-location)"
    LINK: str = "designation of method used to link branches in parallelism data"
    LOADER: str = "loading procedure to use for input data"
    RANDOM_SEED: str = "nonnegative integer seed for randomized processes"
    RESULTS_DIRECTORY: str = "directory in which other designated output files will be contained"
    SCORING_MODE: str = "parallelism metric used for evaluation"
    STRATUM_COUNT: str = "number of tag layers to account for in input data"
    TAGSET: str = "designation of method used to label individual branches in parallelism data"
    TEST_FILENAME: str = "filename prefix for test results"
    TRAINING_FILENAME: str = "filename prefix for per-epoch outputted model training results"
    VALIDATION_FILENAME: str = "filename prefix for per-epoch outputted model validation results"
