# Binary Vulnerability Detection With Machine Learning
This repository contains the work-in-progress code for my method of using machine learning to identify vulnerabilities in binaries.
This work stemmed from a research project during my graduate CS degree coursework.
To avoid any potenial issues, I won't name any specific degree program or institution - not that you couldn't probably figure it out.
Suffice to say, if you steal my work for that (or any) course, the professor is going to know.
Don't be an idiot, and also, don't be a dick.

## Current status
Classifier: Naive Bayes

Accuracy: ~22%

Problems: Horrific overfitting

## How does it work?
First off, it's important to know that at the moment, it doesn't.
This approach has very poor results, but I believe it is the beginning of something that can be refined to achieve usable results for guiding further vulnerability analysis (see [Why do this](#why-do-this)).

The approach utilizes static binary analysis with a disassembler to featurize a binary, such that each observation (row) is a basic block, and each observation's features are the opcodes observed in that basic block.
The observations are labeled with the CWE category present in that binary.
Currently the approach uses a Naive Bayes classifier to train and test, but this is going to be changed in the future as I believe I have exhausted the potential of this classifier and still not achieved particularly good results.

The classifier is trained on binaries from the [CGC dataset](https://github.com/trailofbits/cb-multios).

## What's in here?
Well obviously the python files run the machine learning magic.
Some of them, anyway.
The lion's share of the machine learning magic is done in `cs613.py` (see my earlier comment about being able to guess the course).
The other python scripts are mostly runners, except for `confusion_matrix.py` which is a helper for generating confusion matrix graphs.
Then, there's a handful of CSV files.

- `bin_test.csv` - a binary-feature dataset of some CGC binaries
- `evt_test.csv` - a counted-feature dataset of some CGC binaries
- `cb-labels.csv` - a CSV that maps CWE categories to some CGC binaries

## Why do this?
There are a lot of reasons to do this.
Vulnerability identification in binaries is done by developers to locate bugs and fix them before they're exploited in the wild.
It is also done by researchers as the first step in exploit development.
Creating a machine learning tool that can identify vulnerabilities in code removes the need for a human to analyze the code, increasing the number of bugs found vs time.

Admittedly, I don't think this approach will ever achieve very high accuracy (but you never know!).
That doesn't mean it's not useful, though!
There are plenty of automated tools that can be fed program addresses and attempt to validate the existence of a vulnerability.
For example, a symbolic execution tool (such as [Angr](https://github.com/angr/angr) or [S2E](https://s2e.systems/)) can dynamically force a binary to execute to a given program address and check for program crashes.
You can imagine giving the output of this machine learning tool to those kinds of tools and receiving a list of confirmed vulnerabilities, all without having to do anything yourself, and all in a matter of minutes, rather than the days that fuzzing or symbolic execution can take.
