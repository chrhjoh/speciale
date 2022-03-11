import os
import re

def parse_kernel_output(filename):
    """ Parses a file of kernel and inserts information about the peptide which was tested"""
    REG_EX = r"/P\d_(\w+)_\w+$"
    get_peptide = re.compile(REG_EX)
    parsed = []
    with open(filename, 'r') as fh:
        for line in fh:
            line = line.strip()
            # Find peptide information from filename input
            if line.startswith("#"):
                pattern = get_peptide.search(line)
                if pattern:
                    peptide = pattern.group(1)
            else:
                # All lines with information are subset and peptide is added
                line = line.split()[1:7]
                line.append(peptide)
                parsed.append(line)
    return parsed

def write_output(output, outfile):
    with open(outfile, "w") as fh:
        for line in output:
            print(",".join(line), file=fh)
    
def main():

    OUT_DIR = "/Users/christianjohansen/Desktop/speciale/baseline/out"
    RES_FILE = os.path.join(OUT_DIR, "baseline_all.out")
    OUT_FILE = os.path.join(OUT_DIR, "baseline_all.csv")
    output = parse_kernel_output(RES_FILE)
    write_output(output, OUT_FILE)

    
    
if __name__ == "__main__":
    main()