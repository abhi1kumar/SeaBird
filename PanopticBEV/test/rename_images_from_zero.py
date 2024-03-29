"""
    Sample Run:
    python .py
"""
import os, sys
sys.path.append(os.getcwd())

import argparse
import glob
import numpy as np
np.set_printoptions   (precision= 2, suppress= True)

def execute(command, print_flag= False):
    if print_flag:
        print(command)
    os.system(command)

#================================================================
# Main starts here
#================================================================
parser = argparse.ArgumentParser(description='implementation of DEVIANT')
parser.add_argument('--folder', type=str, default = "images/qualitative/seabird_kitti360_demo", help='evaluate model on validation set')
parser.add_argument('--style',  type=str, default= "link", help= 'link/copy/move')
args   = parser.parse_args()

folder = args.folder
style  = args.style

files_list   = sorted(glob.glob(folder + "/*.png"))
num_files    = len(files_list)
# out_folder = os.path.dirname(files_list[0])
out_folder   = folder + "_renamed"
os.makedirs(out_folder, exist_ok= True)
CWD = os.getcwd()

# Reorder file list
start_ind = 55382
end_ind   = 55500
diff_old  = 264
diff_new  = end_ind - start_ind + 1
new_files_list  = []
for i in range(num_files):
    new_files_list.append(None)
for i, fpath in enumerate(files_list):
    t = int(os.path.basename(fpath).replace(".png", ""))
    if t < start_ind:
        new_files_list[i+diff_new] = fpath
    elif t >= start_ind and t <= end_ind:
        new_files_list[i-diff_old] = fpath
    else:
        new_files_list[i]      = fpath
files_list = new_files_list

print("Found {} files in {}".format(num_files, folder))
print("Output folder=    {}".format(out_folder))

for i, inp_path in enumerate(files_list):
    inp_path   = os.path.join(CWD, inp_path)
    out_id     = "{:06d}".format(i)
    out_path   = os.path.join(CWD, out_folder, str(out_id) + ".png")

    if style == "copy":
        command = "cp " + inp_path + " " + out_path
    elif style == "move":
        command = "mv " + inp_path + " " + out_path
    else:
        command = "ln -sfn " + inp_path + " " + out_path
    execute(command)
    if (i+1) % 100 == 0 or (i+1) == num_files:
        print("{} images done".format(i+1))