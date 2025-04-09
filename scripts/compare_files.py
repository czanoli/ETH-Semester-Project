# File paths (replace with your actual file names if needed)
file1_path = "/home/tatiana/chris-sem-prj/ETH-Semester-Project/debug/dino_vitl14-reg_layer18/mspd_errors_dinoentireImage.txt"
file2_path = '/home/tatiana/chris-sem-prj/ETH-Semester-Project/debug/dino_vitl14-reg_layer18/mspd_errors_dinoOnlySegmented.txt'
output_path = '/home/tatiana/chris-sem-prj/ETH-Semester-Project/debug/dino_vitl14-reg_layer18/dino_mspd_comparisonOutput.txt'

# Read error values from both files
with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
    errors1 = [float(line.strip()) for line in f1]
    errors2 = [float(line.strip()) for line in f2]

if len(errors1) != len(errors2):
    raise ValueError("Files must have the same number of lines")

# Compare errors and prepare output lines
comparison_results = []
for e1, e2 in zip(errors1, errors2):
    if e2 < e1:
        comparison_results.append("second file has lower error")
    elif e1 < e2:
        comparison_results.append("first file has lower error")
    else:
        comparison_results.append("both have equal error")

# Write results to output file
with open(output_path, 'w') as out_file:
    out_file.write("\n".join(comparison_results))

# Compute and print mean errors
mean1 = sum(errors1) / len(errors1)
mean2 = sum(errors2) / len(errors2)

print(f"Mean error of first file: {mean1:.6f}")
print(f"Mean error of second file: {mean2:.6f}")
