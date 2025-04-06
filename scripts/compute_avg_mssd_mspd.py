import os

def read_and_average(filename):
    try:
        with open(filename, 'r') as f:
            numbers = [float(line.strip()) for line in f if line.strip()]
        return sum(numbers) / len(numbers) if numbers else 0.0
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

for i in range(12):
    dec_label = f"dec{i}"
    mssd_file = f"mssd_errors_{dec_label}.txt"
    mspd_file = f"mspd_errors_{dec_label}.txt"

    avg_mssd = read_and_average(mssd_file)
    avg_mspd = read_and_average(mspd_file)

    print(f"---- {dec_label} ----")
    if avg_mssd is not None and avg_mspd is not None:
        print(f"avg MSSD: {avg_mssd:.4f}, avg MSPD: {avg_mspd:.4f}\n")
    else:
        print("Missing one or both files.\n")
