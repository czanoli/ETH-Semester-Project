import os

base_fold = "debug/ViTBase_BaseDecoder"
ndecblks = 9   # 12 for ViTLarge_BaseDecoder | 9 for ViTBase_Small_Decoder

def read_and_average(filename):
    try:
        with open(filename, 'r') as f:
            numbers = [float(line.strip()) for line in f if line.strip()]
        return sum(numbers) / len(numbers) if numbers else 0.0
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None

for i in range(ndecblks):
    dec_label = f"dec{i}"
    mssd_file = f"{base_fold}/mssd_errors_{dec_label}.txt"
    mspd_file = f"{base_fold}/mspd_errors_{dec_label}.txt"

    avg_mssd = read_and_average(mssd_file)
    avg_mspd = read_and_average(mspd_file)

    print(f"---- {dec_label} ----")
    if avg_mssd is not None and avg_mspd is not None:
        print(f"avg MSSD: {avg_mssd:.4f}, avg MSPD: {avg_mspd:.4f}\n")
    else:
        print("Missing one or both files.\n")
