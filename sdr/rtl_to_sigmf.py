"""Turn a raw SDR recording into a SigMF recording (cf32_le) with one labelled signal.
Example: python rtl_to_sigmf.py fm.cu8 --rate 2.4e6 --freq 100e6 --offset 200e3 --bw 200e3 --label WBFM --author Kagan --hw "RTL-SDR Blog V4" --comment "gain 20 dB"
"""
import argparse, datetime, os
import numpy as np
from sigmf import SigMFFile

p = argparse.ArgumentParser()
p.add_argument("infile")                                  # .cu8 from rtl_sdr, or .cf32 from GNU Radio File Sink
p.add_argument("--in-type", choices=["cu8", "cf32"], default="cu8")
p.add_argument("--rate", type=float, required=True)       # sample rate in samples per second
p.add_argument("--freq", type=float, required=True)       # centre frequency in Hz
p.add_argument("--label", required=True)                  # exactly as in classes.txt
p.add_argument("--bw", type=float, required=True)         # signal bandwidth in Hz
p.add_argument("--offset", type=float, default=0.0)       # signal centre minus tuned centre, Hz
p.add_argument("--author", required=True)
p.add_argument("--hw", required=True)
p.add_argument("--comment", default="")
a = p.parse_args()

if a.in_type == "cu8":                                    # RTL-SDR raw: 0..255, I,Q,I,Q,...
    raw = np.fromfile(a.infile, dtype=np.uint8)
    clipped = np.mean((raw == 0) | (raw == 255)) * 100
    x = (raw.astype(np.float32) - 127.5) / 127.5
    iq = (x[0::2] + 1j * x[1::2]).astype(np.complex64)
else:                                                     # GNU Radio File Sink, complex
    iq = np.fromfile(a.infile, dtype=np.complex64)
    clipped = np.mean(np.maximum(abs(iq.real), abs(iq.imag)) >= 0.99) * 100

base = os.path.splitext(a.infile)[0]
iq.tofile(base + ".sigmf-data")
meta = SigMFFile(data_file=base + ".sigmf-data", global_info={
    "core:datatype": "cf32_le", "core:sample_rate": a.rate,
    "core:author": a.author, "core:hw": a.hw,
    "core:description": f"{a.label} at {a.freq/1e6:.3f} MHz"})
meta.add_capture(0, metadata={"core:frequency": a.freq,
    "core:datetime": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
fc = a.freq + a.offset
meta.add_annotation(0, len(iq), metadata={
    "core:freq_lower_edge": fc - a.bw / 2, "core:freq_upper_edge": fc + a.bw / 2,
    "core:label": a.label, "core:comment": a.comment})
meta.tofile(base + ".sigmf-meta")
print(f"{len(iq):,} samples = {len(iq)/a.rate:.2f} s, clipped {clipped:.3f} %  ->  {base}.sigmf-meta / .sigmf-data")
if clipped > 0.1: print("WARNING: clipping. Lower the gain and record again.")
