import os
from Bio import SeqIO
import random
from tqdm import tqdm
from ete2 import NCBITaxa


# ----------- #
#  Functions  #
# ----------- #


next_taxon_level = {"species": "genus",
                    "genus": "family",
                    "family": "order",
                    "order": "class",
                    "class": "phylum",
                    "phylum": "kingdom",
                    "kingdom": "domain",
                    "domain": "root"}


def get_desired_ranks(taxid, desired_ranks):
    """ Gets the parent TaxID for a particular Taxon node.

        Args:
            taxid: Taxon Node whose parent TaxID at a given level needs
                   to be determined
            desired_ranks: Parent Taxon level at which Taxon ID must be
                           determined.

        Returns:
            List of TaxIDs at particular taxon level

    """
    ncbi = NCBITaxa()
    lineage = ncbi.get_lineage(taxid)
    names = ncbi.get_taxid_translator(lineage)
    lineage2ranks = ncbi.get_rank(names)
    ranks2lineage = dict((rank, taxid) for (taxid, rank) in
                         lineage2ranks.items())

    return [ranks2lineage.get(rank, '0') for rank in desired_ranks]


def get_parent2children(all_fasta_files, parent_level):
    """ Dictionary with keys as TaxIDs at parent taxon level and
        Value as a list of genome filenames.

        Args:
            all_fasta_files: Path to where source genomes are stored.
            parent_level: Taxon level at which to group genomes.

        Example:
            all_fasta_files -> 1304.fna, 1308.fna ...
            parent_level: genus

            Result:
                    1301: [1304.fna, 1308.fna ...]
                    1228: [1220.fna ...]

        Returns:
            Dictionary of TaxID to list of genomes

    """
    parent2child = dict()
    for fasta_file in os.listdir(all_fasta_files):
        key = get_desired_ranks(int(fasta_file.strip().split('.')[0]),
                                [parent_level])[0]
        if key not in parent2child:
            parent2child[key] = []
        parent2child[key].append(int(fasta_file.strip().split('.')[0]))
    return parent2child


def get_parent2children_next_level(child_dataset_path, parent_level,
                                   directory_for_db_creation):
    """ Dictionary with keys as TaxIDs at parent taxon level and
        Value as a list of TaxIDs one level below parent taxon level.

        Args:
            child_dataset_path: Path to where child TaxIDs can be found
            parent_level: Taxon level at which to group child TaxIDs
            directory_for_db_creation: db_creation directory

        Example:
            child_dataset_path (genus) -> 1301, 1228
            parent_level: family
            directory_for_db_creation: db_creation

            Result:
                    1300: [1301, 1302 ...]
                    1200: [1228 ...]

        Returns:
            Dictionary of Parent TaxID to list of child TaxIDs

    """
    parent2child = dict()
    for directory in os.listdir(os.path.join(child_dataset_path, directory_for_db_creation)):
        if not directory.endswith('.py') and not directory.endswith('.pklz'):
            key = get_desired_ranks(directory, [parent_level])[0]
            if key not in parent2child:
                parent2child[key] = []
            parent2child[key].append(directory)
    return parent2child


def create_from_species(all_fasta_files, output_directory, current_taxon_level,
                        final_taxon_level, fragment_len, train_fraction,
                        train_set=True):
    """ Creates the first dataset from raw genomes at the lowest level -- Species
        Calls create_next_dataset after creating first dataset.

        Args:
            all_fasta_files: Path to genomes
            output_directory: Path to where datasets need to be stored
            current_taxon_level: Current level at which dataset is being created
            final_taxon_level: Stopping criterion. Level at which to end creation
                               of datasets.
            fragment_len: Length of fragment to extract from input sequences.
                          This will be the inputs to classifiers.

                          Note: The parameter fragment_len used in neural nets
                                must be the same as the one provided here.

            train_fraction: Fraction of data to be used for trianing vs testing.

                            Note: In this implementation, training phase uses
                                  all input sequences. This is because whole
                                  genomes are held out for testing.

            train_set: If this is set to true, then database (originial data) is
                       saved.


        Returns: None

    """
    try:
        next_taxon_level[final_taxon_level]
    except:
        return

    if current_taxon_level == next_taxon_level[final_taxon_level]:
        return

    directory_for_ml_training = 'ml_training'
    directory_for_db_creation = 'db_creation'
    directory_for_ml_testing = 'ml_test'

    print "\n Creating dataset at taxonomy level: {} \n".format(current_taxon_level)

    species2subspecies = get_parent2children(all_fasta_files, current_taxon_level)
    new_output_directory = os.path.join(output_directory, current_taxon_level)

    if os.path.exists(new_output_directory):
        return create_next_dataset(new_output_directory, output_directory,
                                   next_taxon_level[current_taxon_level],
                                   final_taxon_level, fragment_len, train_fraction,
                                   train_set)

    skip_length = fragment_len
    start_pos = 0

    for key, value in tqdm(species2subspecies.items()):
        os.makedirs(os.path.join(new_output_directory, directory_for_ml_training,
                    str(key)))
        os.makedirs(os.path.join(new_output_directory, directory_for_ml_testing,
                    str(key)))

        if train_set is True:
            os.makedirs(os.path.join(new_output_directory,
                        directory_for_db_creation, str(key)))

        with open(os.path.join(new_output_directory, directory_for_ml_training,
                  str(key), str(key) + ".fasta"), 'a') as fasta_handle_train,\
            open(os.path.join(new_output_directory, directory_for_ml_training,
                 str(key), str(key) + ".taxid"), 'a') as taxid_handle_train,\
            open(os.path.join(new_output_directory, directory_for_ml_testing,
                 str(key), str(key) + ".fasta"), 'a') as fasta_handle_test,\
            open(os.path.join(new_output_directory, directory_for_ml_testing,
                 str(key), str(key) + ".taxid"), 'a') as taxid_handle_test:

            if train_set is True:
                fasta_db_handle = open(os.path.join(new_output_directory,
                                       directory_for_db_creation, str(key),
                                       str(key) + ".fasta"), 'a')
                taxid_db_handle = open(os.path.join(new_output_directory,
                                       directory_for_db_creation, str(key),
                                       str(key) + ".taxid"), 'a')

            for fasta_file in tqdm(value):
                for fasta_record in SeqIO.parse(os.path.join(all_fasta_files,
                                                str(fasta_file) + ".fna"),
                                                'fasta'):
                    rec_range = range(len(xrange(start_pos,
                                                 len(fasta_record.seq) - fragment_len + 1,
                                                 skip_length)))
                    random.shuffle(rec_range)
                    test_indexes = rec_range[int(train_fraction * len(rec_range)) + 1:]

                    for cur_position in xrange(start_pos,
                                               len(fasta_record.seq) - fragment_len + 1,
                                               skip_length):
                        text = str(fasta_record.seq)[cur_position:
                                                     cur_position + fragment_len]
                        if len(text) != fragment_len:
                            continue

                        fasta_handle_train.write(">" + fasta_record.id + "\n")
                        fasta_handle_train.write(text + "\n")
                        taxid_handle_train.write(str(fasta_file) + "\n")

                        if cur_position/skip_length in test_indexes:
                            fasta_handle_test.write(">" + fasta_record.id + "\n")
                            fasta_handle_test.write(text + "\n")
                            taxid_handle_test.write(str(fasta_file) + "\n")

                    if train_set is True:
                        fasta_db_handle.write(">" + fasta_record.id + "\n")
                        fasta_db_handle.write(str(fasta_record.seq) + "\n")
                        taxid_db_handle.write(str(fasta_file) + "\n")

            if train_set is True:
                fasta_db_handle.close()
                taxid_db_handle.close()

    return create_next_dataset(new_output_directory, output_directory,
                               next_taxon_level[current_taxon_level],
                               final_taxon_level, fragment_len, train_fraction,
                               train_set)


def create_next_dataset(child_dataset_path, output_directory, current_taxon_level,
                        final_taxon_level, fragment_len, train_fraction,
                        train_set=True):
    """ Creates all datasets after the first (species) dataset.
        Recursively calls itself until dataset at final_taxon_level is created.

        Args:
            child_dataset_path: Path to previously created dataset.
            output_directory: Path to where datasets need to be stored
            current_taxon_level: Current level at which dataset is being created
            final_taxon_level: Stopping criterion. Level at which to end creation
                               of datasets.
            fragment_len: Length of fragment to extract from input sequences.
                          This will be the inputs to classifiers.

                          Note: The parameter fragment_len used in neural nets
                                must be the same as the one provided here.

            train_fraction: Fraction of data to be used for trianing vs testing.

                            Note: In this implementation, training phase uses
                                  all input sequences. This is because whole
                                  genomes are held out for testing.

            train_set: If this is set to true, then database (originial data) is
                       saved.


        Returns: None

    """
    try:
        next_taxon_level[final_taxon_level]
    except:
        return

    if current_taxon_level == next_taxon_level[final_taxon_level]:
        return

    directory_for_ml_training = 'ml_training'
    directory_for_db_creation = 'db_creation'
    directory_for_ml_testing = 'ml_test'

    print "\n \n Creating dataset at taxonomy level: {} \n".format(current_taxon_level)

    parent2child = get_parent2children_next_level(child_dataset_path,
                                                  current_taxon_level,
                                                  directory_for_db_creation)

    new_output_directory = os.path.join(output_directory, current_taxon_level)

    if os.path.exists(new_output_directory):
        return create_next_dataset(new_output_directory, output_directory,
                                   next_taxon_level[current_taxon_level],
                                   final_taxon_level, fragment_len, train_fraction,
                                   train_set)

    skip_length = fragment_len
    start_pos = 0

    for key, value in tqdm(parent2child.items()):
        os.makedirs(os.path.join(new_output_directory, directory_for_ml_training,
                                 str(key)))
        os.makedirs(os.path.join(new_output_directory, directory_for_ml_testing,
                                 str(key)))

        if train_set is True:
            os.makedirs(os.path.join(new_output_directory,
                        directory_for_db_creation, str(key)))

        with open(os.path.join(new_output_directory, directory_for_ml_training,
                  str(key), str(key) + ".fasta"), 'a') as fasta_handle_train,\
            open(os.path.join(new_output_directory, directory_for_ml_training,
                 str(key), str(key) + ".taxid"), 'a') as taxid_handle_train,\
            open(os.path.join(new_output_directory, directory_for_ml_testing,
                              str(key), str(key) + ".fasta"), 'a') as fasta_handle_test,\
            open(os.path.join(new_output_directory, directory_for_ml_testing,
                              str(key), str(key) + ".taxid"), 'a') as taxid_handle_test:

            if train_set is True:
                fasta_db_handle = open(os.path.join(new_output_directory,
                                                    directory_for_db_creation,
                                                    str(key), str(key) + ".fasta"), 'a')
                taxid_db_handle = open(os.path.join(new_output_directory,
                                                    directory_for_db_creation,
                                                    str(key), str(key) + ".taxid"), 'a')

            for directory in tqdm(value):
                current_directory = os.path.join(child_dataset_path,
                                                 directory_for_db_creation,
                                                 directory)

                for fasta_file in os.listdir(current_directory):
                    if fasta_file.endswith('.taxid'):
                        continue

                    for fasta_record in SeqIO.parse(os.path.join(current_directory,
                                                    fasta_file), 'fasta'):
                        rec_range = range(len(xrange(start_pos,
                                                     len(fasta_record.seq) - fragment_len + 1,
                                                     skip_length)))
                        random.shuffle(rec_range)
                        test_indexes = rec_range[int(train_fraction * len(rec_range)) + 1:]

                        for cur_position in xrange(start_pos,
                                                   len(fasta_record.seq) - fragment_len + 1,
                                                   skip_length):
                            text = str(fasta_record.seq)[cur_position:
                                                         cur_position + fragment_len]
                            if len(text) != fragment_len:
                                continue
                            fasta_handle_train.write(">" + fasta_record.id + "\n")
                            fasta_handle_train.write(text + "\n")
                            taxid_handle_train.write(directory + "\n")

                            if cur_position/skip_length in test_indexes:
                                fasta_handle_test.write(">" + fasta_record.id + "\n")
                                fasta_handle_test.write(text + "\n")
                                taxid_handle_test.write(directory + "\n")

                        if train_set is True:
                            fasta_db_handle.write(">" + fasta_record.id + "\n")
                            fasta_db_handle.write(str(fasta_record.seq) + "\n")
                            taxid_db_handle.write(directory + "\n")

            if train_set is True:
                fasta_db_handle.close()
                taxid_db_handle.close()

    return create_next_dataset(new_output_directory, output_directory,
                               next_taxon_level[current_taxon_level],
                               final_taxon_level, fragment_len, train_fraction,
                               train_set)
