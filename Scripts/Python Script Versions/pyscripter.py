import os

#os.path.join(DATA_DIR, "feature_set-retail.csv")
#related_nbs = ['squaredplus.ipynb', 'untitled.ipynb' ]

#todo: if no list passed in, just convert all nb files in the dir

from os import listdir
from os.path import isfile, join


def nb_to_py(*args):    
    if len(args) == 0:
        mypath = os.getcwd()
        related_nbs = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and '.ipynb' in f)]
    else:
        related_nbs = args[0]
    
    for nb in related_nbs:      
        if ' ' in nb:
            print("Could not convert " + nb + ". Please remove spaces" )
        else: 
            os.system('jupyter nbconvert --to script ' + nb) 
    
    related_nbs[:] = [nb[0:-6] for nb in related_nbs if ' ' not in nb]
    print(related_nbs)
    
    return related_nbs

def import_scripts(scripts):
    for nb in scripts:
        try:
            __import__(nb)
            print(nb + " not imported.")
        except ModuleNotFoundError as error:
            # Output expected ImportErrors.
            print("Missing module in " + nb + ", so it was not imported.")
        except:
            print(nb + " was not imported.")
            