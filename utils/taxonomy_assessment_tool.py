import os
import sys
from ete2 import NCBITaxa
import getopt
from tqdm import tqdm
import numpy as np


tax_order = ["species", "genus", "family", "order", "class", "phylum"]
ncbi = NCBITaxa()


def get_paths_upto_parent(ground_truth_taxid, predicted_taxid):
    """ Pruned Predicted TaxonID upto Taxon level defined by
        Ground Truth TaxonID.

        Note:
            If Predicted TaxonID's Taxon level is lower than
            Ground Truth TaxonID, then Predicted TaxonID is
            not pruned.

        Args:
            ground_truth_taxid: Ground Truth TaxonID.
            predicted_taxid: Predicted TaxonID.

        Returns:
            New Predicted TaxonID.

    """
    gt_rank = ncbi.get_rank([ground_truth_taxid])[ground_truth_taxid]
    predicted_ranks_dict = ncbi.get_rank(ncbi.get_lineage(predicted_taxid))
    taxid_split = -1

    for taxid in predicted_ranks_dict.keys():
        if predicted_ranks_dict[taxid] == gt_rank:
            taxid_split = taxid
            return taxid
    if taxid_split == -1:
        for i in tax_order:
            for taxid in predicted_ranks_dict.keys():
                if predicted_ranks_dict[taxid] == i:
                    taxid_split = taxid
                    return taxid


def calculate_scores(ground_truth_labels, predicted_labels):
    """ Computes Precision, Sensitivity and F1 Scores given
        ground truth and predicted labels.

        Args:
            ground_truth_labels: List of Ground Truth TaxonIDs.
            predicted_labels: List of Predicted TaxonIDs.

        Returns:
            Three floats - precision, sensitivity and f1score

    """

    tuples = zip(ground_truth_labels, predicted_labels)

    # Precision, Sensitivity and F1 Score of all samples in test dataset

    dataset_precision, dataset_sensitivity, dataset_f1score = [], [], []

    unique_labels = np.unique(ground_truth_labels)

    # Used to determine per TaxID Precision, Sensitivity and F1 Score
    f1_dict = dict()
    precision_dict = dict()
    sensitivity_dict = dict()

    for i in unique_labels:
        f1_dict[i] = []
        precision_dict[i] = []
        sensitivity_dict[i] = []

    for pair in tqdm(tuples):
        # pair[1] is the predicted TaxID.
        # KRAKEN (modified by me) outputs 0 if a sample is Unclassified

        if pair[1] == 0:
            dataset_precision.append(0)
            dataset_sensitivity.append(0)
            dataset_f1score.append(0)

            f1_dict[pair[0]].append(0)
            precision_dict[pair[0]].append(0)
            sensitivity_dict[pair[0]].append(0)
            continue

        ground_truth_path = ncbi.get_lineage(pair[0])
        predicted_path = ncbi.get_lineage(pair[1])

        predicted_from_root = list(predicted_path)
        ground_truth_from_root = list(ground_truth_path)

        # Sometimes predicted TaxID level may be lower than ground truth TaxID level
        # Example: Predicted TaxID - 1234 (Sub-Species)
        #        Ground Truth TaxID - 2345 (Species)
        # 'get_paths_upto_parent' acts as the pruning function to ensure fair evaluation

        new_predicted_label = get_paths_upto_parent(pair[0], pair[1])
        new_predicted_rank = ncbi.get_rank([new_predicted_label])[new_predicted_label]

        predicted_from_root = predicted_from_root[:predicted_from_root.index(new_predicted_label) + 1]

        ans = []
        for i, j in zip(predicted_from_root, ground_truth_from_root):
            if i == j:
                ans.append(i)
            else:
                break

        # Refer Ver.0.2.0 of F1, Precision & Sensitivity Calculation
        tpCount = len(ans)
        fpCount = len(predicted_from_root) - len(ans)
        fnCount = len(ground_truth_from_root) - len(ans)

        precision = float(tpCount) / (float(tpCount) + float(fpCount))
        sensitivity = float(tpCount) / (float(tpCount) + float(fnCount))
        f1score = (2 * precision * sensitivity) / (precision + sensitivity)

        f1_dict[pair[0]].append(f1score)
        precision_dict[pair[0]].append(precision)
        sensitivity_dict[pair[0]].append(sensitivity)

        dataset_precision.append(precision)
        dataset_sensitivity.append(sensitivity)
        dataset_f1score.append(f1score)

    p = sum(dataset_precision) / len(dataset_precision)
    s = sum(dataset_sensitivity) / len(dataset_sensitivity)
    f1 = sum(dataset_f1score) / len(dataset_f1score)

    return [p, s, f1]


def taxa_assessment(path_to_ground_truth_labels, path_to_predicted_labels):
    """ Driver function to perform Taxonomy Evaluation.

        Args:
            path_to_ground_truth_labels: File containing Ground Truth TaxonIDs. One per line.
            path_to_predicted_labels: File containing Predicted TaxonIDs. One per line.

        Returns:
            None

    """

    # Barebones check if both paths exist
    if os.path.exists(path_to_predicted_labels) and os.path.exists(path_to_ground_truth_labels):
        ground_truth_labels = [int(line.rstrip('\n')) for line in open(path_to_ground_truth_labels)]
        predicted_labels = [int(line.rstrip('\n')) for line in open(path_to_predicted_labels)]
        [p, s, f1] = calculate_scores(ground_truth_labels, predicted_labels)
        print "Preicision: {0:.2f}".format(p*100)
        print "Sensitivity: {0:.2f}".format(s*100)
        print "F1 Score: {0:.2f}".format(f1*100)
    else:
        print "Ensure that both files exist and they have the same number of lines!"


def usage_message():
    print "----------------------------------------------------------------------------------------------------------------------------------"
    print "This program calculates the F1 Scores, Precision and Sensitivity using v.0.2.0 scoring sceme \n"
    print "!!! It assumes that there is one integer per line, and that both (predicted and ground truth) files the have same number of lines."
    print "---------------------------------------------------------------------------------------------------------------------------------- \n"
    print "Usage: taxonomy_assessment_tool.py [-g path to ground truth labels file] -p [path to predicted labels file] \n"
    sys.exit()


def main(init_args):
    if(len(init_args) < 4):
        usage_message()

    optlist, args = getopt.getopt(init_args, 'g:p:')

    for o, v in optlist:
        if(o == '-g'):
            path_to_ground_truth_label = v
        elif(o == "-p"):
            path_to_predicted_label = v

    taxa_assessment(path_to_ground_truth_label, path_to_predicted_label)


if __name__ == "__main__":
        main(sys.argv[1:])
