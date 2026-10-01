"""Roofline helpers: how close is a kernel to the GPU's memory-bandwidth ceiling?

All our tasks are memory-bound, so achieved bandwidth / peak bandwidth is the
honest "how good can this possibly get" metric.
"""
# Peak DRAM bandwidth in GB/s (vendor spec sheets). Order matters: more specific names first.
PEAK_GBPS = [
    ("H200", 4800), ("H100 NVL", 3900), ("H100 SXM", 3350), ("H100 80GB HBM3", 3350), ("H100 PCIe", 2000), ("H100", 3350),
    ("A100-SXM4-80GB", 2039), ("A100 80GB PCIe", 1935), ("A100-PCIE-80GB", 1935), ("A100-SXM4-40GB", 1555),
    ("A100-PCIE-40GB", 1555), ("A100", 1555),
    ("L40S", 864), ("A40", 696), ("L40", 864), ("A6000", 768), ("V100", 900), ("P100", 732),
    ("RTX PRO 6000", 1792),
]


def peak_gbps(gpu_name):
    if not gpu_name:
        return None
    for key, bw in PEAK_GBPS:
        if key.lower() in gpu_name.lower():
            return bw
    return None


def bandwidth_stats(nbytes, ms, gpu_name):
    gbps = nbytes / (ms * 1e-3) / 1e9
    peak = peak_gbps(gpu_name)
    return {"bytes_moved": nbytes, "gbps": gbps, "pct_peak_bw": (100 * gbps / peak) if peak else None}
