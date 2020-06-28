import os
import shutil

def copy_file(sources, destination):
    for source in sources:
        destination_temp = "/".join([destination, source.split("/")[-1]])
        print(source)
        shutil.copyfile(source, destination_temp)

def filter_file(source, sequence_length):

    file_indices = list(map(str, list(range(sequence_length * 2, sequence_length * 2 + 5))))
    files_to_copy = []
    has_gro = False
    for file in os.listdir(source):
        if file[-6:-4] in file_indices:
            if file[-3:] == "gro":
                if has_gro:
                    continue
                else:
                    files_to_copy.append("/".join([source, file]))
                    has_gro = True
            elif file[-3:] == "xtc":
                files_to_copy.append("/".join([source, file]))
    return files_to_copy

if __name__ == "__main__":

    print("\nNOTE: The program assumes that the number of neutral replicas is 5.")
    print("The file format of trajectory should be .xtc and topology should be .gro.\n")
    curr_path = os.getcwd()

    source_path_s1 = input("\nEnter the directory where trajectory (xtc) and topology (gro) files are located for s1\n")
    source_path_s2 = input("\nEnter the directory where trajectory (xtc) and topology (gro) files are located for s2\n")
    sequence_length = int(input("\nEnter the sequence length\n"))

    files_to_copy = filter_file(source_path_s1, sequence_length)
    os.makedirs(curr_path + "/trajectory/s1")
    copy_file(files_to_copy, (curr_path + "/trajectory/s1"))

    os.makedirs(curr_path + "/trajectory/s2")
    files_to_copy = filter_file(source_path_s2, sequence_length)
    copy_file(files_to_copy, (curr_path + "/trajectory/s2"))

    while True:
        cluster = input("Enter 1 for submission on cluster, 0 otherwise")
        if cluster == "0":
            os.system("python dPCA.py --traj1 trajectory/s1 --traj2 trajectory/s2")
            break
        elif cluster == "1":
            os.system("sbatch submit.job")
