import pickle
from os import listdir, makedirs, mkdir
from os.path import isdir, join

def save_var(var, var_name, out_path):
    out_path_and_fn = join(
        out_path,
        f"{var_name}.pkl",
    )
    with open(out_path_and_fn, "wb") as fout:
        pickle.dump(var, fout)

def create_results_dir(out_dir_home):

    if not isdir(out_dir_home):
        mkdir(out_dir_home)

    d = [int(file) for file in listdir(out_dir_home) if isdir(join(out_dir_home, file)) and file.isdigit()]
    if len(d) > 0:
        n = max(d) + 1
    else:
        n = 0

    out_dir = f"{out_dir_home}/{str(n)}"
    makedirs(out_dir)

    return out_dir