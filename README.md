# NLP-Relation-Extraction-Classification
We classify the relation between head and tail entity using a Naive Bayes model based on bag-of-words (BoW) features.

## Data
Input data: Two csv files: train.csv and test.csv, adapted from the [FewRel](https://www.aclweb.org/anthology/D18-1514) dataset.
  Each csv file contains 5 columns: 
  |Column| Example| Description|
  | ---- | ------ | ---------- |
  |row_id|435|The unique row id.|
  |tokens|Trapped and Deceived is a 1994 television film directed by Robert Iscove .|The tokenized sentence, separated by a single space.|
  |relation|director|The correct relation (original label).|
  head_pos|0 1 2|Position of the head entity (Trapped and Deceived). Indices start with 0 and are separated by a single space.|
  |tail_pos|11 12|Position of the tail entity (Robert Iscove).|

## Tasks
### Task 1: Classify Text
  - We wrote a program that trains a Naive Bayes classifier for the 4-class classification problem at hand. It processes texts and classify them as belonging to one of the following classes: publisher, director, performer, and characters. The NB classifier is implemented from scratch.
  - The program should trains the model with the training file, and print the training accuracy (using 3-fold cross validation). After training the model, the program takes the test file as input, makes predictions using the model, writes the predictions to files, and prints the accuracy on the test set.

### Task 2: Improve the Classifier
  - We did one round of improvements based on our error analysis. 

### Output
The output should be a csv file containing the following 3 columns in the same order: original_label, output_label, and row_id.

# Report and Results
Further details and results can be found [here](https://github.com/Leen-Alzebdeh/NLP-Relation-Extraction-Classification/blob/main/REPORT.md)

# Contributors

Leen Alzebdeh  @Leen-Alzebdeh

Sukhnoor Khehra @Sukhnoor-K

# Resources Consulted

- Chapter 4 of the Speech and Language Processing textbook.
- Used ChatGPT to help write the top_mi_selection and calculate_mutual_information functions.
- Used Github Copilot

## Libraries

* `main.py L:[line]` used `[Library Name]` for [reason].
* `main.py L:[5, 15]` used `argparse` for parsing command line arguments.
* `main.py L:[75, 173]` used `csv` to read anf write csv files.
* `main.py L[191] used` `random` to shuffle training data.
* `main.py L[132, 137]` used `math` to find logs of probabilities.
* `main.py L[56, 57]` used `statistics` to find the mean and standard deviation of the training accuracy.
* `main.py L[103, 104, 117, 119, 150, 207]` used `collections` to use defaultdict.
  
# Instructions to execute code

Ensure Python is installed, as well as the Python Standard Library.

**To run the improved mode, add the flag -i.**

Example usage: use the following command in the current directory.

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv -i` for the improved model.

`python3 src/main.py --train data/train.csv --test data/test.csv --output output/test.csv` for the original model.

