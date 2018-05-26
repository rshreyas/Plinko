import os
import sys
import getopt
import subprocess
import seaborn as sns


def opal_predictor(level, taxid, path_to_opal, path_to_opal_model_store):
    models_in_level = os.listdir(os.path.join(path_to_opal_model_store, level))
    kmer_characterized_models = [i for i in models_in_level if i.startswith(taxid)]
    model_store = path_to_opal_model_store

    for model in kmer_characterized_models:
        kmer_val = model.split("_")[-1]
        path_to_predict_directory = os.path.join(".", "opal_predict_output",
                                                 level, model)
        path_to_testfiles = os.path.join(".", level, "ml_test", taxid)
        path_to_opal_model_store = os.path.join(model_store, level, model)
        subprocess.check_call([os.path.join(path_to_opal, "opal.py"), "predict",
                              "-k", kmer_val, path_to_opal_model_store,
                               path_to_testfiles, path_to_predict_directory])


def usage_message():
    print "Usage: opal_predictor.py [-l level in tree] [-t taxid] [-p path to opal.py] [-r model store]"
    sys.exit()


def main(init_args):
    if(len(init_args) < 8):
        usage_message()
    optlist, args = getopt.getopt(init_args, 'l:t:p:r:')
    for o, v in optlist:
        if(o == '-l'):
            level = v
        elif(o == "-t"):
            taxid = v
        elif(o == "-p"):
            path_to_opal = v
        elif(o == "-r"):
            path_to_opal_model_store = v

    opal_predictor(level, taxid, path_to_opal, path_to_opal_model_store)


if __name__ == "__main__":
    main(sys.argv[1:])
