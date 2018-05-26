import os
import shutil
from ete2 import NCBITaxa

all_fasta_files = "all_fasta_files"

ncbi = NCBITaxa()


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
    ranks2lineage = dict((rank, taxid) for (taxid, rank) in lineage2ranks.items())

    return [ranks2lineage.get(rank, '0') for rank in desired_ranks]


def find_genomes_under_taxid(taxid, level):
    """ Assuming all genomes are found in a directory ("all_fasta_files"),
        all genomes under a TaxID at a Taxonomy level are fetched and
        stored in a directory with name "all_fasta_files_<TAXID>"

        Args:
            taxid: All genomes under this TaxID must be extracted
            level: Taxonomy level of taxid

        Returns:
            None

    """
    if not os.path.exists(os.path.join(os.getcwd(),
                          all_fasta_files + "_" + str(taxid))):
        os.makedirs(os.path.join(os.getcwd(), all_fasta_files + "_" + str(taxid)))
    for fasta in os.listdir(os.path.join(all_fasta_files)):
        if get_desired_ranks(int(fasta.split('.')[0]), [level])[0] == taxid:
            shutil.copy(os.path.join(all_fasta_files, fasta),
                        os.path.join(os.getcwd(),
                        all_fasta_files + "_" + str(taxid)))


if __name__ == "__main__":
    find_genomes_under_taxid(1639, "species")
