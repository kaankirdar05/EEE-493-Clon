import numpy as np
from modes import *
rng = np.random.default_rng(0)
for h in ["8D4840D6202CC371C32CE0576098", "8D40621D58C382D690C8AC2863A7", "8D40621D58C386435CC412692AD6"]:
    print(h, remainder(hex2bits(h)))
for fs in (2, 4, 10):
    ok = 0; n = 300
    for i in range(n):
        b = make_df17(rng); iq, s = synth_frame(fs, 80, b, rng)
        iq = iq * rng.uniform(.1, 5)
        r = decode_modes_ex(iq, fs * 1e6, decoder="auto")
        good = [x for x in r if x['crc_ok'] and x['hex'] == bits2hex(b)]
        ok += len(good) > 0
        if not good and i < 3: print(fs, i, len(r), [(x['index'], x['crc_ok']) for x in r][:5], s)
    print(fs, "noiseless decode", ok / n)
