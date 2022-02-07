def load_sequences(file_path) -> dict:
    headers = []
    sequences = []
    seq_lst = []
    with open(file_path, 'r') as fh:
        for line in fh:
            if line.startswith('>'):
                headers.append(line[1:].strip())

                if len(seq_lst) != 0:
                    sequences.append("".join(seq_lst))

                seq_lst = []

            else:
                seq_lst.append(line.strip())
    sequences.append("".join(seq_lst))

    return dict(zip(headers, sequences))

def write_fasta(sequences, filename):
    with open(filename, "w") as fh:
        for id in sequences:
            fh.write(">"+id+"\n")
            for j in range(0, len(sequences[id]), 60):
                fh.write(sequences[id][j:j+60]+"\n")