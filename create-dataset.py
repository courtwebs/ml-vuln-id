import os
import csv
import numpy
from collections import defaultdict
import binaryninja as binja

def load_csv(filename):
    data = []

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            data.append(row)

    return data

"""
From a CSV with two columns, this function returns each row 
where the last column only contains one value. It is intended 
to take a spreadsheet that has the directory of a CB and the 
CWE(s) associated with that CB. For creating our dataset, we 
only want CBs that have a single CWE associated with them, to 
make labeling/training/testing simpler.
"""
def get_single_label_samples(csvfile):
    binaries_to_ignore = ["EternalPass"]
    d = load_csv(csvfile)
    result = []

    for row in d[1:]:
        if row[0] not in binaries_to_ignore:
            cwes = row[1].split(", ")

            if len(cwes) == 1:
                if cwes[0] != '':
                    result.append([row[0], int(cwes[0]), row[2]])

    return result

def print_data_stats(data):
    print("Number of binaries:")
    print(data.shape[0])
    print("Number of unique CWEs:")
    print(len(numpy.unique(data[:,-1:])))

    cwe_counts = defaultdict(int)

    for cwe in data[:,-1:]:
        cwe_counts[cwe[0]] += 1

    print("Number of binaries for each CVE:")
    for cwe, count in cwe_counts.items():
        print("CWE-" + str(cwe) + ": " + str(count))

def get_binary_block_opcodes(bv, filename, cwe):
    print("Featurizing binary '" + str(filename) + "'")
    features = []

    for func in bv.functions:
        for block in func:
            b = []
            for insn, l in block:
                b.append(str(insn[0]).strip())

        features.append([b, filename, cwe])

    return features

def featurize_binaries(data):
    for i in range(data.shape[0]):
        dirname = data[i][0]
        filename = data[i][2]
        cwe = data[i][1]
        target = challenge_dir + os.sep + dirname + os.sep + filename
        print("Loading binary '" + str(target) + "'")
        bv = binja.BinaryViewType["ELF"].open(target)
        bv.update_analysis_and_wait()
        featurized_binaries.extend(get_binary_block_opcodes(bv, filename, cwe))

    return featurized_binaries

def format_blocks_for_csv(featurized_binaries):
    all_opcodes = set()

    # Figure out what the unique set of opcodes is
    for block in featurized_binaries:
        opcodes = set(block[0])
        all_opcodes = all_opcodes.union(opcodes)

    # Convert to a list to preserve ordering. It doesn't particularly matter 
    # what the order is, only that it is consistent every time we iterate 
    # through the list.
    all_opcodes = list(all_opcodes)

    header = []
    header.extend(all_opcodes)
    header.append("binary")
    header.append("cwe")

    dataset = []
    dataset.append(header)

    for block in featurized_binaries:
        block_opcodes = block[0]
        binary_name = block[1]
        cwe = block[2]
        row = []

        for opcode in all_opcodes:
            if opcode in block_opcodes:
                row.append('1')
            else:
                row.append('0')

        row.append(binary_name)
        row.append(cwe)
        dataset.append(row)

    return dataset

def write_dataset_to_file(dataset, filename):
    with open(filename, 'wb') as f:
        for row in dataset:
            f.write(",".join(row))
            f.write("\n")

if __name__ == '__main__':
    challenge_dir = "/home/user/challenges"
    data = get_single_label_samples("/home/user/cb-labels.csv")
#    data = get_single_label_samples("/home/user/less-cb-labels.csv")
    data = numpy.array(data)
    print_data_stats(data)
    featurized_binaries = []

    print("\n---\n")

    featurized_binaries = featurize_binaries(data)
    dataset = format_blocks_for_csv(featurized_binaries)
    write_dataset_to_file(dataset, "test.csv")
