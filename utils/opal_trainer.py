import pandas as pd
import numpy as np
from ete2 import NCBITaxa
import itertools
import os
import sys
import getopt
import subprocess


parent_level_mappings = {"species": "genus",
                         "genus": "family",
                         "family": "order",
                         "order": "class",
                         "class": "phylum",
                         "phylum": "superkingdom"}


df = pd.read_table("kmer_characterization_830_genomes.txt",
                   delimiter='\t', header=None)
df_columns = df.columns.tolist()
df = df[df_columns[0:-1]]
df.sort_values(by=[3], ascending=False).head()


def get_desired_ranks(taxid, desired_ranks):
        ncbi = NCBITaxa()
        lineage = ncbi.get_lineage(taxid)
        names = ncbi.get_taxid_translator(lineage)
        lineage2ranks = ncbi.get_rank(names)
        ranks2lineage = dict((rank, taxid) for (taxid, rank) in lineage2ranks.items())
        return [ranks2lineage.get(rank, '0') for rank in desired_ranks]


def get_name(taxid):
    ncbi = NCBITaxa()
    names = ncbi.get_taxid_translator([taxid])
    return names[taxid]


def get_parent2genome(list_of_child_taxids, parent_level="genus"):
        '''
        Returns a dictionary containing parent taxID : [child taxID] mappings.
        #---------------------------------------------------------------------#
        parent_level taxID: int
        genome taxID: int
        #---------------------------------------------------------------------#
        '''

        parent2genome = dict()

        for taxid in list_of_child_taxids:
            if taxid <= 0:
                continue
            key = get_desired_ranks(taxid, [parent_level])[0]
            if key not in parent2genome:
                parent2genome[key] = []
            parent2genome[key].append(taxid)
        return parent2genome


def kmer_characterization():
    characterization_dictionary = dict()
    for level in ["phylum", "class", "order", "family", "genus", "species"]:
        curr_level = df[df[0] == level]
        all_unique_ = np.unique(curr_level[[1, 2]].values)

        grouping = get_parent2genome(all_unique_, parent_level_mappings[level])
        groupings_remastered = dict()

        for key in grouping.keys():
            '''
            Only characterize at TaxIDs that have more than one genome within them.
            '''
            if not groupings_remastered.has_key(key) and len(grouping[key]) > 1:
                groupings_remastered[key] = []

            if len(grouping[key]) > 1:
                groupings_remastered[key].extend(grouping[key])

        means_for_curr_level, max_for_curr_level, min_for_curr_level = [], [], []

        for current_group in groupings_remastered.keys():
            permutations = list(itertools.combinations(groupings_remastered[current_group], 2))
            characterization_dictionary[current_group] = []

            for tuples in permutations:
                if len(df[(df[1] == tuples[0]) & (df[2] == tuples[1])]) > 0:
                    characterization_dictionary[current_group].extend(
                        df[(df[1] == tuples[0]) & (df[2] == tuples[1])][3])
                elif len(df[(df[1] == tuples[1]) & (df[2] == tuples[0])]) > 0:
                    characterization_dictionary[current_group].extend(
                        df[(df[1] == tuples[1]) & (df[2] == tuples[0])][3])

            if len(characterization_dictionary[current_group]) > 0:
                curr_mean = sum(characterization_dictionary[current_group]) / \
                    len(characterization_dictionary[current_group])
                curr_max = max(characterization_dictionary[current_group])
                curr_min = min(characterization_dictionary[current_group])

                means_for_curr_level.append(curr_mean)
                max_for_curr_level.append(curr_max)
                min_for_curr_level.append(curr_min)

    return characterization_dictionary


def opal_trainer(level, taxid, path_to_opal, characterization):
    # CHANGE THIS
    characterization_ = list(range(8, 16))
    print "Characterization: {}".format(characterization_)

    for kmer_val in characterization_:
        #CHANGE THIS
        row_weight = kmer_val
        assert kmer_val % row_weight == 0
        if kmer_val % row_weight != 0:
            print "Skipping k value {} as row weight condition not met".format(kmer_val)
            continue
        path_to_training_samples = os.path.join(".", level, "db_creation", str(taxid))
        path_to_opal_model_store = os.path.join(".", "opal_model_store",
                                                level,
                                                str(taxid) + "_" + str(row_weight) + "_" + str(kmer_val))
        fragment_length = '100'
        coverage = '1'
        number_of_hashes = '2'

        subprocess.call([os.path.join(path_to_opal, "opal.py"),
                         "train", "-k", str(kmer_val),
                         "-l", fragment_length,
                         "-c", coverage,
                         "--row-weight", str(row_weight),
                         "--num-hash", number_of_hashes,
                         path_to_training_samples, path_to_opal_model_store])


def usage_message():
    print "Usage: opal_trainer.py [-l level in tree] [-t taxid] [-p path to opal.py]"
    sys.exit()


def main(init_args):
    if(len(init_args) < 6):
        usage_message()
    optlist, args = getopt.getopt(init_args, 'l:t:p:')
    for o, v in optlist:
        if(o == '-l'):
            level = v
        elif(o == "-t"):
            taxid = int(v)
        elif(o == "-p"):
            path_to_opal = v

    characterization = kmer_characterization()
    opal_trainer(level, taxid, path_to_opal, characterization)


if __name__ == "__main__":
    main(sys.argv[1:])
