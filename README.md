# Intro-RPD
**Authors**: 
* Stephen Bothwell, Theresa Crnkovich, Hildegund Müller, and David Chiang (University of Notre Dame)
* Justin DeBenedetto (Villanova University)

**Maintainer**: Stephen Bothwell

## Summary

This is a repository for the EMNLP 2023 paper, **"Introducing Rhetorical Parallelism Detection: 
A New Task with Datasets, Metrics, and Baselines."**
It contains pertinent code, results, and tools for public use. 
All code was produced in Python 3.9, and it has not been tested in versions outside of it.

## Contents

First, this repository contains data relevant to the reproduction of results in our work. 
We created a directory structure to mimic or simplify that which was used in our own development process. 
We add information about data splits in each of the datasets' `splits` subdirectories under `data`.

Second, we provide the results of various studies in our paper in `results`. 
The `bootstrapping` subdirectory contains the ASP dataset's bootstrapped inter-annotator agreement results. 
The `error-analysis` subdirectory holds data tagged for the sake of the paper's error analysis; it is formatted with a tagged word per line, 
and the labels applied to each parallelism are described in more detail in the accompanying PDF file. 
The `tagging` subdirectory contains CSV files and tabular data listing all results obtained for models deemed "best" by our hyperparameter search procedure 
across all of our metrics.

Third, this repository contains a variety of CLI-based tools which can be used in replicating the results of our work 
or in expanding upon that work. At the current time, we provide a basic description of each tool as well as a display of its CLI. 
All such descriptions and CLIs are provided below.

### Analysis

The following three tools are used for analyzing data. 
The first, `bio_analyzer.py`, examines parallelism data relative to the BIO tagging schemes proposed in our work. 
The second, `parallelism_analyzer.py`, computes a variety of frequency-based and distributional statistics regarding parallelisms.
The third, `results_analyzer.py`, produces a small set of significance testing and visualization outputs for model performance data on our four metrics.

To use the `bio_analyzer.py` and `parallelism_analyzer.py` items, 
place your data in the desired location (*e.g.*, the `data` directory or its subdirectories). To use the `results_analyzer.py`,
create a CSV file containing results and place it in the desired location--for example, the `results` directory or its subdirectories. 
Two results files have been provided already containing results from our work: 
`aggregated_asp_results.csv` and `aggregated_pse_results.csv`. 
For an easier-to-read version of these results, see the accompanying PDF: `Intro-RPD - Supplementary Results.pdf`.

#### BIO Analyzer

```
>>> python bio_analyzer.py -h
usage: bio_analyzer.py [-h] [--collection-format {document,section}] [--link {bd,td}] [--loader LOADER] [--mappings [{} ...]] [--output-filepath OUTPUT_FILEPATH]     
                       [--save-formats [{graphml,heatmap,latex} ...]] [--save-frequency-style {counts,ratios}] [--save-order [{B,I,C,J,O,M,E,B-X,<START>,<STOP>} ...]]
                       [--save-path SAVE_PATH] [--stratum-count STRATUM_COUNT] [--tagset {bio,bioe,bioj,biom,bioje,biomj,biome,biomje}]                               
                       input_filepath                                                                                                                                 
                                                                                                                                                                      
positional arguments:                                                                                                                                                 
  input_filepath        existing input directory containing compatible XML parallelism data                                                                           

optional arguments:
  -h, --help            show this help message and exit
  --collection-format {document,section}
                        designation of textual units in parallelism data
  --link {bd,td}        designation of method used to link branches in parallelism data
  --loader LOADER       loading procedure to use for input data
  --mappings [{} ...]   tag mappings to be used to combine discrete nodes into common classes for the AdjacencyList
  --output-filepath OUTPUT_FILEPATH
                        path to output file (sans extension) where the resulting AdjacencyList's statistics will be stored
  --save-formats [{graphml,heatmap,latex} ...]
                        formats in which information about the AdjacencyList data should be saved
  --save-frequency-style {counts,ratios}
                        style in which information about edge counts should be used and/or saved
  --save-order [{B,I,C,J,O,M,E,B-X,<START>,<STOP>} ...]
                        order in which heatmap nodes should be depicted
  --save-path SAVE_PATH
                        location where specially-formatted AdjacencyList data should be saved
  --stratum-count STRATUM_COUNT
                        number of tag layers to account for in input data
  --tagset {bio,bioe,bioj,biom,bioje,biomj,biome,biomje}
                        designation of method used to label individual branches in parallelism data
```

#### Parallelism Analyzer

```
>>> python parallelism_analyzer.py -h 
usage: parallelism_analyzer.py [-h] [--aggregate | --no-aggregate] [--loader {asp,pse-i}] [--visualize | --no-visualize] input_directory output_directory
                                                                                                                                                         
positional arguments:                                                                                                                                    
  input_directory       existing directory path containing XML parallelism data                                                                          
  output_directory      directory path to contain statistical information for input data                                                                 
                                                                                                                                                         
optional arguments:                                                                                                                                      
  -h, --help            show this help message and exit                                                                                                  
  --aggregate, --no-aggregate
                        flag to determine whether aggregate stats will be computed (default: False)
  --loader {asp,pse-i}  loading procedure to use for input data
  --visualize, --no-visualize
                        flag to determine whether matplotlib-based visualizations will be generated (default: False)
```

#### Result Analyzer

```
>>> python result_analyzer.py -h
usage: result_analyzer.py [-h] [--alpha ALPHA] --analysis-type {friedman,box} [--criteria {embedding,encoder,tagset,link} [{embedding,encoder,tagset,link} ...]]
                          [--input-file INPUT_FILE] [--output-filepath OUTPUT_FILEPATH] [--scoring-mode {mwo,mbawo,mpbm,epm}] [--split {training,validation,optimization,test}]     

optional arguments:
  -h, --help            show this help message and exit
  --alpha ALPHA         offset to determine p-value (e.g., an alpha of 0.95 generates a p-value of 0.05)
  --analysis-type {friedman,box}
                        type of analysis to perform over results data
  --criteria {embedding,encoder,tagset,link} [{embedding,encoder,tagset,link} ...]
                        selection of filter criteria for data
  --input-file INPUT_FILE
                        CSV input filepath containing results information
  --output-filepath OUTPUT_FILEPATH
                        filepath used to store outputs; currently only used for boxplot creation
  --scoring-mode {mwo,mbawo,mpbm,epm}
                        parallelism metric used for evaluation
  --split {training,validation,optimization,test}
                        split of data examined for results analysis
```

### Data

The following five tools are used for generating, examining, and manipulating data. 
The first, `bootstrapper.py`, was used in our bootstrapping experiments 
to approximate inter-annotator agreement over our entire dataset 
when one annotator provided us with a sample from the data.
The second, `brat_converter.py`, converts data from `brat` (Stenetorp *et al.* 2012) 
in the form of text (`.txt`) and annotation (`.ann`) files to XML. 
The third, `data_splitter.py`, contains our procedure for dividing parallelism data 
into approximately balanced splits (*e.g.*, training, validation, optimization, and test). 
The fourth, `pse_format_converter.py`, combines new `brat` annotation data and the original XML-based, sentence-level parallelism annotations 
(Song *et al.* 2016) into a new word-level format. 
The fifth, `result_gatherer.py`, collects a results as produced by other systems (*i.e.*, `neural_tagger.py`) into a single CSV file.

To apply these tools as in our work, the following resources have been provided:
* For working with `brat_converter.py`, we provide our original `brat` annotations in our ASP data repository, which is available [here](https://github.com/Mythologos/Augustinian-Sermon-Parallelisms) in the `original` subdirectory.
* For working with `bootstrapper.py`, we furnish the data used for our inter-annotator agreement studies in the ASP repository under the `agreement-study` subdirectory.
* For working with `data_splitter.py`, we recommend obtaining either the [ASP](https://github.com/Mythologos/Augustinian-Sermon-Parallelisms) or [PSE-I](https://github.com/Mythologos/Paibi-Student-Essays) datasets. 
For the ASP dataset, the untokenized XML data can be used with this tool; meanwhile, for the PSE-I dataset, the data available under `augmented/full` can be used.
* For working with `pse_format_converter.py`, we present both the original PSE data and our `brat` annotations in the [PSE](https://github.com/Mythologos/Paibi-Student-Essays) repository. We also note any errata that the script does not handle, since a very small number of items were altered manually.
* For working with `result_gatherer.py`, we currently do not have files available for this purpose. 
However, such files can be generated by using the `neural_tagger.py` file and capturing evaluation results outputted to the console in a file.

#### Bootstrapping

``` 
>>> python bootstrapper.py -h   
usage: bootstrapper.py [-h] [--alpha ALPHA] [--cleaners [{conjunctions,interlocks} ...]] [--loader {asp,pse-i}] [--output-file OUTPUT_FILE] [--random-seed RANDOM_SEED]
                       [--sample-count SAMPLE_COUNT] [--sample-percentage SAMPLE_PERCENTAGE] [--scoring-mode {mwo,mbawo,mpbm,epm}]                                     
                       first_filepath second_filepath                                                                                                                  
                                                                                                                                                                       
positional arguments:                                                                                                                                                  
  first_filepath        location of first dataset directory for bootstrapping procedure                                                                                
  second_filepath       location of second dataset directory for bootstrapping procedure                                                                               

optional arguments:
  -h, --help            show this help message and exit
  --alpha ALPHA         value determining confidence interval
  --cleaners [{conjunctions,interlocks} ...]
                        preprocessors to clean up parallelism data before matching
  --loader {asp,pse-i}  loading procedure to use for input data
  --output-file OUTPUT_FILE
                        output filepath for bootstrapping results
  --random-seed RANDOM_SEED
                        nonnegative integer seed for randomized processes
  --sample-count SAMPLE_COUNT
                        number of samples to perform for bootstrap estimate
  --sample-percentage SAMPLE_PERCENTAGE
                        percentage of samples, relative to original number of samples, to extract per bootstrap sample
  --scoring-mode {mwo,mbawo,mpbm,epm}
                        parallelism metric used for evaluation
```

#### ASP: Brat Conversion

```
>>> python brat_converter.py -h 
usage: brat_converter.py [-h] [--capitalization | --no-capitalization] [--punctuation | --no-punctuation] [--punctuation-strategy {preserve,exclude}]
                         [--sectioning | --no-sectioning] [--truncation | --no-truncation]                                                           
                         input_filepath output_filepath                                                                                              
                                                                                                                                                     
positional arguments:                                                                                                                                
  input_filepath        directory containing text and annotation files from brat                                                                     
  output_filepath       directory to contain XML outputs built from brat's files                                                                     

optional arguments:
  -h, --help            show this help message and exit
  --capitalization, --no-capitalization
                        flag determining whether the data will be capitalized or not (default: True)
  --punctuation, --no-punctuation
                        flag determining whether the data will contain punctuation or not (default: True)
  --punctuation-strategy {preserve,exclude}
                        procedure for determining how punctuation will interact with parallelism tags;the 'preserve' strategy will keep punctuation as the .ann file labels it,     
                        whereas the 'exclude' strategy will bump punctuation out of the edges of any branch
  --sectioning, --no-sectioning
                        flag determining whether section breaks will be included or not (default: True)
  --truncation, --no-truncation
                        flag determining whether text files without annotation data will be represented in the final output or not (default: True)
```

#### Data Splitting

``` 
>>> python data_splitter.py -h 
usage: data_splitter.py [-h] [--match-directory MATCH_DIRECTORY] [--sets [SETS ...]] [--ratios [RATIOS ...]] [--loader {asp,pse-i}] [--stratum-count STRATUM_COUNT]
                        input_directory output_directory

positional arguments:
  input_directory       an existing directory containing XML parallelism data
  output_directory      an existing directory where specified data splits will be stored

optional arguments:
  -h, --help            show this help message and exit
  --match-directory MATCH_DIRECTORY
                        an existing directory where XML parallelism data of the same kind has already been split
  --sets [SETS ...]     names of the splits
  --ratios [RATIOS ...]
                        ratios (relative to 1) of the data per split
  --loader {asp,pse-i}  loading procedure to use for input data
  --stratum-count STRATUM_COUNT
                        number of tag layers to account for in input data
```

#### PSE-I Conversion

``` 
>>> python pse_format_converter.py -h          
usage: pse_format_converter.py [-h] [--annotation-directory ANNOTATION_DIRECTORY] [--output-directory OUTPUT_DIRECTORY] [--pse-directory PSE_DIRECTORY]

optional arguments:
  -h, --help            show this help message and exit
  --annotation-directory ANNOTATION_DIRECTORY
                        directory containing word-level brat parallelism annotations
  --output-directory OUTPUT_DIRECTORY
                        directory to contain output PSE-I data
  --pse-directory PSE_DIRECTORY
                        directory containing original XML-style sentence-level parallelism data
```

#### Result Gathering

```
>>> python result_gatherer.py -h      
usage: result_gatherer.py [-h] [--file-regex FILE_REGEX] [--file-type {txt}] [--output-file OUTPUT_FILE] [--subdirectory-regex SUBDIRECTORY_REGEX] directories [directories ...]

positional arguments:
  directories           list of directories to traverse for gathering result data

optional arguments:
  -h, --help            show this help message and exit
  --file-regex FILE_REGEX
                        Python-style regular expression used for permitting the inclusion of files in result compilation
  --file-type {txt}     type of file to gather result data from
  --output-file OUTPUT_FILE
                        filepath to (possibly nonexistent) result compilation file
  --subdirectory-regex SUBDIRECTORY_REGEX
                        Python-style regular expression used for excluding subdirectories in result compilation
```

### Learning

The following two tools are useful for training models for rhetorical parallelism detection. 
The first, `hyperparameter_trial_generator.py`, can create a variety of scripts for executing hyperparameter searches 
across different model architectures. The second, `neural_tagger.py`, allows for the training and evaluation of the Encoder-CRF 
system with our tagging schemes and metrics. For the latter tool, we present three interfaces in accordance with the main interface of the script 
and its two mode-specific scripts.

#### Hyperparameter Trial Generator

``` 
>>> python hyperparameter_trial_generator.py -h 
usage: hyperparameter_trial_generator.py [-h]
                                         [--varied-hyperparameters [{activation,bidirectional,blender,collection-format,dataset,dropout,epochs,frozen-embeddings,heads,layers,learni
ng-rate,lemmatization,link,optimizer,patience,random-seed,replacement-probability,replacement-strategy,scoring-mode,stratum-count,tagset,weight-decay,embeddings,embedding-dim-first
,hidden-dim-first} ...]]
                                         [--specified-hyperparameters [SPECIFIED_HYPERPARAMETERS ...]] [--added-constraints [{word-embedding,word-dim-compression} ...]]
                                         [--model-location MODEL_LOCATION] [--model-name MODEL_NAME] [--results-directory RESULTS_DIRECTORY]
                                         [--training-filename TRAINING_FILENAME] [--validation-filename VALIDATION_FILENAME] [--test-filename TEST_FILENAME]
                                         [--test-partition TEST_PARTITION] [--seed SEED] [--trials TRIALS] [--trial-start-offset TRIAL_START_OFFSET]
                                         [--output-filename OUTPUT_FILENAME] [--output-format {text,bash}]
                                         {chinese-bert,latin-bert,learned,latin-learned-subword,word} {identity,lstm,transformer}

positional arguments:
  {chinese-bert,latin-bert,learned,latin-learned-subword,word}
                        fixed model embedding type for all trials
  {identity,lstm,transformer}
                        fixed model encoder type for all trials

optional arguments:
  -h, --help            show this help message and exit
  --varied-hyperparameters [{activation,bidirectional,blender,collection-format,dataset,dropout,epochs,frozen-embeddings,heads,layers,learning-rate,lemmatization,link,optimizer,pat
ience,random-seed,replacement-probability,replacement-strategy,scoring-mode,stratum-count,tagset,weight-decay,embeddings,embedding-dim-first,hidden-dim-first} ...]
                        hyperparameters which will be varied across all trials
  --specified-hyperparameters [SPECIFIED_HYPERPARAMETERS ...]
                        hyperparameter names and values which will be fixed across all trials
  --added-constraints [{word-embedding,word-dim-compression} ...]
                        constraints to apply to hyperparameter trial generation
  --model-location MODEL_LOCATION
                        directory in which a model is or will be contained
  --model-name MODEL_NAME
                        filename prefix for a model at a designated location (see model-location)
  --results-directory RESULTS_DIRECTORY
                        directory in which other designated output files will be contained
  --training-filename TRAINING_FILENAME
                        filename prefix for per-epoch outputted model training results
  --validation-filename VALIDATION_FILENAME
                        filename prefix for per-epoch outputted model validation results
  --test-filename TEST_FILENAME
                        filename prefix for test results
  --test-partition TEST_PARTITION
                        name of data partition used for evaluation scripts
  --seed SEED           nonnegative integer seed for randomized processes
  --trials TRIALS       number of hyperparameter trials to generate
  --trial-start-offset TRIAL_START_OFFSET
                        starting number for hyperparameter trial generation
  --output-filename OUTPUT_FILENAME
                        output filename prefix for trial files
  --output-format {text,bash}
                        output filetype for trial files
```

#### Neural Tagger

When applying some Encoder-CRF models, external resources may be needed. 
For instance, Latin word2vec embeddings (Burns *et al.* 2021), Latin BERT (Bamman and Burns 2020), and Chinese BERT with whole word masking (Cui *et al.* 2021) were all used.

``` 
>>> python neural_tagger.py -h                  
usage: neural_tagger.py [-h] {train,evaluate} ...                                                    
                                                                                                     
optional arguments:                                                                                  
  -h, --help        show this help message and exit                                                  
                                                                                                     
mode:                                                                                                
  {train,evaluate}  indication of whether a model is being newly trained or both loaded and evaluated
```

``` 
>>> python neural_tagger.py train -h 
usage: neural_tagger.py train [-h] [--collection-format {document,section}] [--dataset DATASET] [--data-splits [DATA_SPLITS ...]] [--evaluation-partition EVALUATION_PARTITION]
                              [--lemmatization | --no-lemmatization] [--link {bd,td}] [--model-location MODEL_LOCATION] [--model-name [MODEL_NAME]]
                              [--output-directory [OUTPUT_DIRECTORY]] [--print-style {all,checkpoint,none}] [--random-seed RANDOM_SEED] [--results-directory [RESULTS_DIRECTORY]]   
                              [--result-display-count RESULT_DISPLAY_COUNT] [--scoring-mode {mwo,mbawo,mpbm,epm}] [--stratum-count STRATUM_COUNT]
                              [--tagset {bio,bioe,bioj,biom,bioje,biomj,biome,biomje}] [--test-filename [TEST_FILENAME]] [--tqdm | --no-tqdm] [--activation-function {relu,gelu}]   
                              [--blender {identity,mean,sum,take-first}] [--dropout DROPOUT] [--embedding-filepath [EMBEDDING_FILEPATH]] [--epochs EPOCHS]
                              [--frozen-embeddings | --no-frozen-embeddings] [--heads [HEADS]] [--hidden-size HIDDEN_SIZE] [--input-size INPUT_SIZE] [--layers [LAYERS]]
                              [--lr [LR]] [--optimizer [OPTIMIZER]] [--patience PATIENCE] [--pretrained-filepath PRETRAINED_FILEPATH]
                              [--replacement-probability [REPLACEMENT_PROBABILITY]] [--replacement-strategy {any,singleton,none}] [--tokenizer-filepath TOKENIZER_FILEPATH]
                              [--training-filename [TRAINING_FILENAME]] [--training-partition TRAINING_PARTITION] [--validation-filename [VALIDATION_FILENAME]]
                              [--weight-decay [WEIGHT_DECAY]] [--visualize [{precision,recall,f1,duration,loss} ...]] [--visualization-directory VISUALIZATION_DIRECTORY]
                              {chinese-bert,latin-bert,learned,latin-learned-subword,word} {identity,lstm,transformer}

optional arguments:
  -h, --help            show this help message and exit

Required Arguments:
  {chinese-bert,latin-bert,learned,latin-learned-subword,word}
                        Encoder-CRF embedding type
  {identity,lstm,transformer}
                        Encoder-CRF encoder type

Common Optional Arguments:
  --collection-format {document,section}
                        designation of textual units in parallelism data
  --dataset DATASET     name of dataset to be used
  --data-splits [DATA_SPLITS ...]
                        names of splits available for the given dataset
  --evaluation-partition EVALUATION_PARTITION
                        name of data partition used for evaluating the model; it is the validation set during training and the test set during testing
  --lemmatization, --no-lemmatization
                        a flag determining whether input tokens are lemmatized (before subword tokenization, if applicable) (default: False)
  --link {bd,td}        designation of method used to link branches in parallelism data
  --model-location MODEL_LOCATION
                        directory in which a model is or will be contained
  --model-name [MODEL_NAME]
                        filename prefix for a model at a designated location (see model-location)
  --output-directory [OUTPUT_DIRECTORY]
                        directory path to which actual tagged model evaluation results will be written
  --print-style {all,checkpoint,none}
                        determines what statements are printed to the console during execution
  --random-seed RANDOM_SEED
                        nonnegative integer seed for randomized processes
  --results-directory [RESULTS_DIRECTORY]
                        directory in which other designated output files will be contained
  --result-display-count RESULT_DISPLAY_COUNT
                        number of nonempty outputs to print or write per epoch
  --scoring-mode {mwo,mbawo,mpbm,epm}
                        parallelism metric used for evaluation
  --stratum-count STRATUM_COUNT
                        number of tag layers to account for in input data
  --tagset {bio,bioe,bioj,biom,bioje,biomj,biome,biomje}
                        designation of method used to label individual branches in parallelism data
  --test-filename [TEST_FILENAME]
                        filename prefix for test results
  --tqdm, --no-tqdm     flag for displaying tqdm-based iterations during processing (default: True)

Training-Specific Arguments:
  --activation-function {relu,gelu}, --af {relu,gelu}
                        activation function used in feedforward layers of a Transformer encoder
  --blender {identity,mean,sum,take-first}
                        blending function used in Encoder-CRF architecture to combine subword embeddings
  --dropout DROPOUT     decimal representing the probability of dropout for neurons in the Transformer encoder
  --embedding-filepath [EMBEDDING_FILEPATH]
                        path to location of relevant word embeddings
  --epochs EPOCHS       maximum number of epochs for training; must be included if patience is not
  --frozen-embeddings, --no-frozen-embeddings
                        a flag determining whether pretrained BERT-based embeddings will be subject to continued training or not (default: True)
  --heads [HEADS], --num-heads [HEADS]
                        number of attention heads per Transformer encoder layer
  --hidden-size HIDDEN_SIZE
                        size of the relevant encoder hidden state
  --input-size INPUT_SIZE, --embedding-size INPUT_SIZE
                        size of the relevant embedding state
  --layers [LAYERS], --num-layers [LAYERS]
                        number of encoder layers
  --lr [LR], --learning-rate [LR]
                        value representing the learning rate used by an optimizer during training
  --optimizer [OPTIMIZER]
                        optimization algorithm used during training
  --patience PATIENCE   maximum number of epochs without validation set improvement before early stopping; must be included if epochs is not
  --pretrained-filepath PRETRAINED_FILEPATH
                        path to location of relevant pretrained BERT model
  --replacement-probability [REPLACEMENT_PROBABILITY]
                        probability that some token will be replaced with <UNK> during training for models with word-level embeddings
  --replacement-strategy {any,singleton,none}
                        strategy by which tokens are deemed viable to be replaced with <UNK> during training for models with word-level embeddings
  --tokenizer-filepath TOKENIZER_FILEPATH
                        filepath to subword tokenizer, if applicable
  --training-filename [TRAINING_FILENAME]
                        filename prefix for per-epoch outputted model training results
  --training-partition TRAINING_PARTITION
                        named split of the data used for training
  --validation-filename [VALIDATION_FILENAME]
                        filename prefix for per-epoch outputted model validation results
  --weight-decay [WEIGHT_DECAY]
                        numeric value for optimizer weight decay
  --visualize [{precision,recall,f1,duration,loss} ...]
                        names of visualization to generate on the basis of training performance
  --visualization-directory VISUALIZATION_DIRECTORY
                        directory path where visualizations created will be stored
```

```
>>> python neural_tagger.py evaluate -h 
usage: neural_tagger.py evaluate [-h] [--collection-format {document,section}] [--dataset DATASET] [--data-splits [DATA_SPLITS ...]] [--evaluation-partition EVALUATION_PARTITION]
                                 [--lemmatization | --no-lemmatization] [--link {bd,td}] [--model-location MODEL_LOCATION] [--model-name [MODEL_NAME]]
                                 [--output-directory [OUTPUT_DIRECTORY]] [--print-style {all,checkpoint,none}] [--random-seed RANDOM_SEED]
                                 [--results-directory [RESULTS_DIRECTORY]] [--result-display-count RESULT_DISPLAY_COUNT] [--scoring-mode {mwo,mbawo,mpbm,epm}]
                                 [--stratum-count STRATUM_COUNT] [--tagset {bio,bioe,bioj,biom,bioje,biomj,biome,biomje}] [--test-filename [TEST_FILENAME]] [--tqdm | --no-tqdm]    
                                 {chinese-bert,latin-bert,learned,latin-learned-subword,word} {identity,lstm,transformer}

optional arguments:
  -h, --help            show this help message and exit

Required Arguments:
  {chinese-bert,latin-bert,learned,latin-learned-subword,word}
                        Encoder-CRF embedding type
  {identity,lstm,transformer}
                        Encoder-CRF encoder type

Common Optional Arguments:
  --collection-format {document,section}
                        designation of textual units in parallelism data
  --dataset DATASET     name of dataset to be used
  --data-splits [DATA_SPLITS ...]
                        names of splits available for the given dataset
  --evaluation-partition EVALUATION_PARTITION
                        name of data partition used for evaluating the model; it is the validation set during training and the test set during testing
  --lemmatization, --no-lemmatization
                        a flag determining whether input tokens are lemmatized (before subword tokenization, if applicable) (default: False)
  --link {bd,td}        designation of method used to link branches in parallelism data
  --model-location MODEL_LOCATION
                        directory in which a model is or will be contained
  --model-name [MODEL_NAME]
                        filename prefix for a model at a designated location (see model-location)
  --output-directory [OUTPUT_DIRECTORY]
                        directory path to which actual tagged model evaluation results will be written
  --print-style {all,checkpoint,none}
                        determines what statements are printed to the console during execution
  --random-seed RANDOM_SEED
                        nonnegative integer seed for randomized processes
  --results-directory [RESULTS_DIRECTORY]
                        directory in which other designated output files will be contained
  --result-display-count RESULT_DISPLAY_COUNT
                        number of nonempty outputs to print or write per epoch
  --scoring-mode {mwo,mbawo,mpbm,epm}
                        parallelism metric used for evaluation
  --stratum-count STRATUM_COUNT
                        number of tag layers to account for in input data
  --tagset {bio,bioe,bioj,biom,bioje,biomj,biome,biomje}
                        designation of method used to label individual branches in parallelism data
  --test-filename [TEST_FILENAME]
                        filename prefix for test results
  --tqdm, --no-tqdm     flag for displaying tqdm-based iterations during processing (default: True)
```

## Contributing

This repository contains code relating to our EMNLP 2023 paper. The code was altered and cleaned before submission to promote usability,
but it is possible that bugs were introduced in the interim. 
If you experience issues in using this code or request more instructions in reproducing our results,
please feel free to submit an issue regarding this.

We do not intend to heavily maintain this code, as it is meant to represent our paper at its time of publication. 
Exceptions may be made if warranted (*e.g.*, there is a bug which prevents the code from being correctly run), 
and we are happy to provide clarifications or assistance in reproducing our results. 

## Citations

To cite this repository, please refer to the following paper:

```
@inproceedings{bothwellIntroducingRPD2023,
    author = {Bothwell, Stephen and DeBenedetto, Justin and Crnkovich, Theresa and M{\"u}ller, Hildegund and Chiang, David},
    title = "Introducing Rhetorical Parallelism Detection: {A} New Task with Datasets, Metrics, and Baselines",
    booktitle = "Proc. EMNLP",
    year = "2023",
    note = "To appear"
}
```

For other works referenced above, see the following:

```
@inproceedings{stenetorpBratWebbasedTool2012,
  title = {{{brat}}: A Web-Based Tool for {{NLP-assisted}} Text Annotation},
  booktitle = {Proceedings of the Demonstrations Session at {{EACL}} 2012},
  author = {Stenetorp, Pontus and Pyysalo, Sampo and Topi{\'c}, Goran and Ohta, Tomoko and Ananiadou, Sophia and Tsujii, Jun'ichi},
  year = {2012},
  month = apr,
  publisher = {{Association for Computational Linguistics}},
  address = {{Avignon, France}}
}

@inproceedings{songLearningIdentifySentence2016,
  title = {Learning to Identify Sentence Parallelism in Student Essays},
  booktitle = {Proceedings of {{COLING}} 2016, the 26th International Conference on Computational Linguistics: {{Technical}} Papers},
  author = {Song, Wei and Liu, Tong and Fu, Ruiji and Liu, Lizhen and Wang, Hanshi and Liu, Ting},
  year = {2016},
  month = dec,
  pages = {794--803},
  publisher = {{The COLING 2016 Organizing Committee}},
  address = {{Osaka, Japan}},
  abstract = {Parallelism is an important rhetorical device. We propose a machine learning approach for automated sentence parallelism identification in student essays. We build an essay dataset with sentence level parallelism annotated. We derive features by combining generalized word alignment strategies and the alignment measures between word sequences. The experimental results show that sentence parallelism can be effectively identified with a F1 score of 82\% at pair-wise level and 72\% at parallelism chunk level.Based on this approach, we automatically identify sentence parallelism in more than 2000 student essays and study the correlation between the use of sentence parallelism and the types and quality of essays.}
}

@inproceedings{burnsProfilingIntertextualityLatin2021,
  title = {Profiling of Intertextuality in {{Latin}} Literature Using Word Embeddings},
  booktitle = {Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: {{Human}} Language Technologies},
  author = {Burns, Patrick J. and Brofos, James A. and Li, Kyle and Chaudhuri, Pramit and Dexter, Joseph P.},
  year = {2021},
  month = jun,
  pages = {4900--4907},
  publisher = {{Association for Computational Linguistics}},
  address = {{Online}},
  doi = {10.18653/v1/2021.naacl-main.389},
  abstract = {Identifying intertextual relationships between authors is of central importance to the study of literature. We report an empirical analysis of intertextuality in classical Latin literature using word embedding models. To enable quantitative evaluation of intertextual search methods, we curate a new dataset of 945 known parallels drawn from traditional scholarship on Latin epic poetry. We train an optimized word2vec model on a large corpus of lemmatized Latin, which achieves state-of-the-art performance for synonym detection and outperforms a widely used lexical method for intertextual search. We then demonstrate that training embeddings on very small corpora can capture salient aspects of literary style and apply this approach to replicate a previous intertextual study of the Roman historian Livy, which relied on hand-crafted stylometric features. Our results advance the development of core computational resources for a major premodern language and highlight a productive avenue for cross-disciplinary collaboration between the study of literature and NLP.}
}

@misc{bammanLatinBERTContextual2020,
  title = {Latin {{BERT}}: {{A}} Contextual Language Model for Classical Philology},
  shorttitle = {Latin {{BERT}}},
  author = {Bamman, David and Burns, Patrick J.},
  year = {2020},
  month = sep,
  eprint = {2009.10053},
  urldate = {2020-09-27},
  abstract = {We present Latin BERT, a contextual language model for the Latin language, trained on 642.7 million words from a variety of sources spanning the Classical era to the 21st century. In a series of case studies, we illustrate the affordances of this language-specific model both for work in natural language processing for Latin and in using computational methods for traditional scholarship: we show that Latin BERT achieves a new state of the art for part-of-speech tagging on all three Universal Dependency datasets for Latin and can be used for predicting missing text (including critical emendations); we create a new dataset for assessing word sense disambiguation for Latin and demonstrate that Latin BERT outperforms static word embeddings; and we show that it can be used for semantically-informed search by querying contextual nearest neighbors. We publicly release trained models to help drive future work in this space.},
  archiveprefix = {arxiv},
  keywords = {Computer Science - Computation and Language},
}

@article{cuiPretrainingWholeWord2021,
  title = {Pre-Training with Whole Word Masking for {{Chinese BERT}}},
  author = {Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing},
  year = {2021},
  journal = {IEEE Transactions on Audio, Speech and Language Processing},
  doi = {10.1109/TASLP.2021.3124365}
}
```
