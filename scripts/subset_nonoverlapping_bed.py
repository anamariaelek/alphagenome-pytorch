import sys

def read_bed(file_path):
    intervals = []
    with open(file_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            intervals.append((chrom, start, end, line.strip()))
    return intervals

def overlap_fraction(a, b):
    # a and b are (chrom, start, end, ...)
    if a[0] != b[0]:
        return 0.0
    start = max(a[1], b[1])
    end = min(a[2], b[2])
    overlap = max(0, end - start)
    length = a[2] - a[1]
    if length == 0:
        return 0.0
    return overlap / length

def main(bed_file, percent):
    intervals = read_bed(bed_file)
    percent = float(percent) / 100.0
    # Sort intervals by chrom, start, end
    intervals.sort(key=lambda x: (x[0], x[1], x[2]))
    keep = []
    prev = None
    for interval in intervals:
        if prev is None:
            keep.append(interval)
            prev = interval
        else:
            if overlap_fraction(interval, prev) > percent:
                continue  # skip this interval
            else:
                keep.append(interval)
                prev = interval
    for interval in keep:
        print(interval[3])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python subset_nonoverlapping_bed.py <input.bed> <percent_overlap>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
