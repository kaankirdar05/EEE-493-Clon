PRACTICE KIT: from a raw SDR recording to training clips (no SDR needed)

Setup once:   pip install numpy scipy matplotlib sigmf h5py

1) Make a fake RTL-SDR recording (a real RTL-SDR would give you the same kind of file):
     python make_practice_recording.py
   -> practice_qpsk.cu8   (raw bytes: I,Q,I,Q,... as numbers 0..255)

2) Raw file -> SigMF (adds the description: rate, frequency, where the signal is, its label):
     python rtl_to_sigmf.py practice_qpsk.cu8 --rate 2.4e6 --freq 150e6 --offset 300e3 --bw 135e3 --label QPSK --author Me --hw "practice file"
   -> practice_qpsk.sigmf-data + practice_qpsk.sigmf-meta   (open the -meta one in Notepad)

3) Look at it:
     python check_sigmf.py practice_qpsk
   -> practice_qpsk_check.png : one block about 135 kHz wide at 150.3 MHz

3b) Measure the true in-band SNR of the signal (the dataset's 'answer key'):
     python measure_snr.py practice_qpsk
   -> QPSK: in-band SNR = 31.4 dB   (with a real generator-off recording: add --noise <name>)

4) SigMF -> HDF5 training clips:
     python sigmf_to_h5.py practice_qpsk --out real_v1.h5
   -> real_v1.h5 : X (N, 1024, 2), Y = class numbers, Z = SNR (unknown here, NaN)

5) Open the clips exactly as you would open RadioML 2018:
     python read_h5.py

With a real RTL-SDR, step 1 becomes one command instead:
     rtl_sdr -f 149700000 -s 2400000 -g 20 -n 2400000 my_recording.cu8
(tune about 300 kHz below the signal; -n 2400000 = 1 second at 2.4 MS/s)
