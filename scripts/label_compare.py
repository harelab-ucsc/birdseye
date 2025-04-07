import os
import difflib
import argparse


parser = argparse.ArgumentParser(description="SLICannotator - aerial data labeling verification tool")
parser.add_argument("reference_file", type=str, help="Reference file, e.g. labels_checked.txt")
parser.add_argument("test_file", type=str, help="Test file, e.g. labels.txt")

args = parser.parse_args()

ref = args.reference_file
tst = args.test_file

def label_reader(filename, _sep=' '):
    print(filename)
    with open(filename, 'r') as f:
        zero_count = 0
        one_count = 0
        test_count = 0
        for idx, line in enumerate(f):
            values = line.strip().split(_sep)  # split the line into two values
            value = values[-1].strip()  # extract the last value and remove trailing whitespace
            if value == '0.0':
                zero_count += 1
            elif value == '1.0':
                one_count += 1
            else:
                test_count += 1
    print(f'  False count: {zero_count}')
    print(f'  True count: {one_count}')
    print(f'  Test count: {test_count}\n')
    return zero_count, one_count, test_count

for filename in [ref,tst]:
    z, o, t = label_reader(filename)

def compare_labels(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        # initialize counters for both files
        zero_count_file1 = 0
        one_count_file1 = 0
        test_count_file1 = 0
        zero_count_file2 = 0
        one_count_file2 = 0
        test_count_file2 = 0

        # initialize counters for differences
        diff_count = 0
        false_pos = 0
        false_neg = 0
        test_pos = 0
        test_neg = 0
        pos_test = 0
        neg_test = 0

        # read lines from both files
        lines_file1 = f1.readlines()
        lines_file2 = f2.readlines()

        # iterate through lines and compare values
        for line1, line2 in zip(lines_file1, lines_file2):
            value1 = line1.strip().split()[-1].strip()
            value2 = line2.strip().split()[-1].strip()

            if value1 != value2:
                diff_count += 1

                if value1 == '1.0' and value2 == '0.0':
                    one_count_file1 += 1
                    zero_count_file2 += 1
                    false_neg += 1

                elif value1 == '0.0' and value2 == '1.0':
                    zero_count_file1 += 1
                    one_count_file2 += 1
                    false_pos += 1

                elif value1 == '1.0' and value2 == 'Test':
                    one_count_file1 += 1
                    test_count_file2 += 1
                    test_pos += 1

                elif value1 == '0.0' and value2 == 'Test':
                    zero_count_file1 += 1
                    test_count_file2 += 1
                    test_neg += 1

                elif value2 == '1.0' and value1 == 'Test':
                    test_count_file1 += 1
                    one_count_file2 += 1
                    pos_test += 1

                elif value2 == '0.0' and value1 == 'Test':
                    test_count_file1 += 1
                    zero_count_file2 += 1
                    neg_test += 1
            elif value1 == '1.0':
                one_count_file1 += 1
                one_count_file2 += 1
            elif value1 == '0.0':
                zero_count_file1 += 1
                zero_count_file2 += 1
            elif value1 == 'Test':
                test_count_file1 += 1
                test_count_file2 += 1

    print(f'Error Rate: {(false_neg+false_pos)/(one_count_file2+zero_count_file2) * 100:.03f}%')
    print(f'  False positive count: {false_pos} (FP Rate: {false_pos/one_count_file2 * 100:.03f}%)')
    print(f'  False negative count: {false_neg} (FN Rate: {false_neg/zero_count_file2 * 100:.03f}%)')
    print()
    try:
        print(f'Test Conversion Rate: {(test_neg+test_pos)/test_count_file2 * 100:.03f}%')
        print(f'  "Test" labels resolved to True: {test_pos} (Test-True Conversion Rate: {test_pos/test_count_file2 * 100:.03f}%)')
        print(f'  "Test" labels resolved to False: {test_neg} (Test-False Conversion Rate: {test_neg/test_count_file2 * 100:.03f}%)')
        print()
        print(f'Revert to "Test" Rate: {(pos_test+neg_test)/(one_count_file2+zero_count_file2) * 100:.03f}%')
        print(f'  True labels switched to "Test": {pos_test} (True-Test Conversion Rate: {pos_test/one_count_file2 * 100:.03f}%)')
        print(f'  False labels switched to "Test": {neg_test} (False-Test Conversion Rate: {neg_test/zero_count_file2 * 100:.03f}%)')
        print()
    except ZeroDivisionError:
        pass
    print(f'Different lines: {diff_count}')

# call function
compare_labels(ref, tst)
