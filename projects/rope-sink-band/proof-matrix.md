| claim | seed | variant | control | result | artifact |
|---|---|---|---|---|---|
| band stability | cal | Qwen3-1.7B calibration | none | B={61,62,63}, k=3, share 0.613 | data/Qwen3-1.7B/band.json |
| band stability | cal2 | Qwen3-1.7B calibration-2 | none | B=[61, 62, 63], Jaccard 1.00 | data/Qwen3-1.7B/calib_cal2.npz |
| location | cal | Qwen3-1.7B | chance C(18,k)/C(64,k) | min index 61: edge a>=21 True, c>=40 True, b>=46 True | data/Qwen3-1.7B/band.json |
| headline | A | Qwen3-1.7B zero band {61,62,63}, all layers | none | cut 0.547 [0.545, 0.550], S 0.477->0.216 | data/Qwen3-1.7B/arms/A__P.npz |
| control: random band | A | Qwen3-1.7B zero [60, 61, 63] | random band (seed 1234) | cut 0.429 | data/Qwen3-1.7B/arms/A__R01.npz |
| control: random band | A | Qwen3-1.7B zero [6, 16, 57] | random band (seed 1234) | cut -0.014 | data/Qwen3-1.7B/arms/A__R02.npz |
| control: random band | A | Qwen3-1.7B zero [7, 33, 50] | random band (seed 1234) | cut -0.017 | data/Qwen3-1.7B/arms/A__R03.npz |
| control: random band | A | Qwen3-1.7B zero [19, 49, 61] | random band (seed 1234) | cut 0.115 | data/Qwen3-1.7B/arms/A__R04.npz |
| control: random band | A | Qwen3-1.7B zero [16, 27, 34] | random band (seed 1234) | cut -0.012 | data/Qwen3-1.7B/arms/A__R05.npz |
| control: random band | A | Qwen3-1.7B zero [40, 53, 55] | random band (seed 1234) | cut 0.032 | data/Qwen3-1.7B/arms/A__R06.npz |
| control: random band | A | Qwen3-1.7B zero [3, 41, 59] | random band (seed 1234) | cut 0.049 | data/Qwen3-1.7B/arms/A__R07.npz |
| control: random band | A | Qwen3-1.7B zero [11, 13, 54] | random band (seed 1234) | cut -0.003 | data/Qwen3-1.7B/arms/A__R08.npz |
| control: random band | A | Qwen3-1.7B zero [3, 10, 29] | random band (seed 1234) | cut -0.003 | data/Qwen3-1.7B/arms/A__R09.npz |
| control: random band | A | Qwen3-1.7B zero [8, 39, 41] | random band (seed 1234) | cut -0.005 | data/Qwen3-1.7B/arms/A__R10.npz |
| control: random band | A | Qwen3-1.7B zero [11, 25, 61] | random band (seed 1234) | cut 0.090 | data/Qwen3-1.7B/arms/A__R11.npz |
| control: random band | A | Qwen3-1.7B zero [0, 12, 33] | random band (seed 1234) | cut 0.001 | data/Qwen3-1.7B/arms/A__R12.npz |
| control: random band | A | Qwen3-1.7B zero [27, 40, 54] | random band (seed 1234) | cut -0.004 | data/Qwen3-1.7B/arms/A__R13.npz |
| control: random band | A | Qwen3-1.7B zero [44, 45, 59] | random band (seed 1234) | cut 0.047 | data/Qwen3-1.7B/arms/A__R14.npz |
| control: random band | A | Qwen3-1.7B zero [22, 49, 62] | random band (seed 1234) | cut 0.200 | data/Qwen3-1.7B/arms/A__R15.npz |
| control: random band | A | Qwen3-1.7B zero [5, 58, 61] | random band (seed 1234) | cut 0.156 | data/Qwen3-1.7B/arms/A__R16.npz |
| control: random band | A | Qwen3-1.7B zero [26, 52, 57] | random band (seed 1234) | cut -0.013 | data/Qwen3-1.7B/arms/A__R17.npz |
| control: random band | A | Qwen3-1.7B zero [53, 54, 62] | random band (seed 1234) | cut 0.198 | data/Qwen3-1.7B/arms/A__R18.npz |
| control: random band | A | Qwen3-1.7B zero [26, 28, 48] | random band (seed 1234) | cut -0.008 | data/Qwen3-1.7B/arms/A__R19.npz |
| control: random band | A | Qwen3-1.7B zero [0, 5, 29] | random band (seed 1234) | cut 0.005 | data/Qwen3-1.7B/arms/A__R20.npz |
| prereg: loss | A | Qwen3-1.7B zero band | median of 20 random bands | dloss +1.582 vs +0.111 nats | data/Qwen3-1.7B/arms/A__P.npz |
| prereg: logit lens | A | Qwen3-1.7B zero band | median of 20 random bands | dlens -0.0200 vs -0.0026 | data/Qwen3-1.7B/arms/A__P.npz |
| licence: temperature | A | Qwen3-1.7B zero band + row STD rescale | temperature | cut 0.523 | data/Qwen3-1.7B/arms/A__T.npz |
| necessity: lead removed | A | Qwen3-1.7B first-token lead of band removed | none | cut 0.571 | data/Qwen3-1.7B/arms/A__GN.npz |
| rival: slowest pairs | A | Qwen3-1.7B zero pairs 56-63 | none | cut 0.885 | data/Qwen3-1.7B/arms/A__LOW8.npz |
| secondary: direct per-layer | A | Qwen3-1.7B band zeroed one layer at a time, upstream clean | random band | cut 0.626 vs random max 0.535 | data/Qwen3-1.7B/arms/A__base.npz |
| headline | B | Qwen3-1.7B zero band {61,62,63}, all layers | none | cut 0.548 [0.546, 0.551], S 0.473->0.214 | data/Qwen3-1.7B/arms/B__P.npz |
| control: random band | B | Qwen3-1.7B zero [60, 61, 63] | random band (seed 1234) | cut 0.428 | data/Qwen3-1.7B/arms/B__R01.npz |
| control: random band | B | Qwen3-1.7B zero [6, 16, 57] | random band (seed 1234) | cut -0.014 | data/Qwen3-1.7B/arms/B__R02.npz |
| control: random band | B | Qwen3-1.7B zero [7, 33, 50] | random band (seed 1234) | cut -0.017 | data/Qwen3-1.7B/arms/B__R03.npz |
| control: random band | B | Qwen3-1.7B zero [19, 49, 61] | random band (seed 1234) | cut 0.116 | data/Qwen3-1.7B/arms/B__R04.npz |
| control: random band | B | Qwen3-1.7B zero [16, 27, 34] | random band (seed 1234) | cut -0.012 | data/Qwen3-1.7B/arms/B__R05.npz |
| control: random band | B | Qwen3-1.7B zero [40, 53, 55] | random band (seed 1234) | cut 0.032 | data/Qwen3-1.7B/arms/B__R06.npz |
| control: random band | B | Qwen3-1.7B zero [3, 41, 59] | random band (seed 1234) | cut 0.049 | data/Qwen3-1.7B/arms/B__R07.npz |
| control: random band | B | Qwen3-1.7B zero [11, 13, 54] | random band (seed 1234) | cut -0.003 | data/Qwen3-1.7B/arms/B__R08.npz |
| control: random band | B | Qwen3-1.7B zero [3, 10, 29] | random band (seed 1234) | cut -0.003 | data/Qwen3-1.7B/arms/B__R09.npz |
| control: random band | B | Qwen3-1.7B zero [8, 39, 41] | random band (seed 1234) | cut -0.004 | data/Qwen3-1.7B/arms/B__R10.npz |
| control: random band | B | Qwen3-1.7B zero [11, 25, 61] | random band (seed 1234) | cut 0.090 | data/Qwen3-1.7B/arms/B__R11.npz |
| control: random band | B | Qwen3-1.7B zero [0, 12, 33] | random band (seed 1234) | cut 0.004 | data/Qwen3-1.7B/arms/B__R12.npz |
| control: random band | B | Qwen3-1.7B zero [27, 40, 54] | random band (seed 1234) | cut -0.005 | data/Qwen3-1.7B/arms/B__R13.npz |
| control: random band | B | Qwen3-1.7B zero [44, 45, 59] | random band (seed 1234) | cut 0.047 | data/Qwen3-1.7B/arms/B__R14.npz |
| control: random band | B | Qwen3-1.7B zero [22, 49, 62] | random band (seed 1234) | cut 0.202 | data/Qwen3-1.7B/arms/B__R15.npz |
| control: random band | B | Qwen3-1.7B zero [5, 58, 61] | random band (seed 1234) | cut 0.156 | data/Qwen3-1.7B/arms/B__R16.npz |
| control: random band | B | Qwen3-1.7B zero [26, 52, 57] | random band (seed 1234) | cut -0.013 | data/Qwen3-1.7B/arms/B__R17.npz |
| control: random band | B | Qwen3-1.7B zero [53, 54, 62] | random band (seed 1234) | cut 0.198 | data/Qwen3-1.7B/arms/B__R18.npz |
| control: random band | B | Qwen3-1.7B zero [26, 28, 48] | random band (seed 1234) | cut -0.008 | data/Qwen3-1.7B/arms/B__R19.npz |
| control: random band | B | Qwen3-1.7B zero [0, 5, 29] | random band (seed 1234) | cut 0.005 | data/Qwen3-1.7B/arms/B__R20.npz |
| prereg: loss | B | Qwen3-1.7B zero band | median of 20 random bands | dloss +1.538 vs +0.100 nats | data/Qwen3-1.7B/arms/B__P.npz |
| prereg: logit lens | B | Qwen3-1.7B zero band | median of 20 random bands | dlens -0.0216 vs -0.0042 | data/Qwen3-1.7B/arms/B__P.npz |
| licence: temperature | B | Qwen3-1.7B zero band + row STD rescale | temperature | cut 0.522 | data/Qwen3-1.7B/arms/B__T.npz |
| necessity: lead removed | B | Qwen3-1.7B first-token lead of band removed | none | cut 0.571 | data/Qwen3-1.7B/arms/B__GN.npz |
| rival: slowest pairs | B | Qwen3-1.7B zero pairs 56-63 | none | cut 0.884 | data/Qwen3-1.7B/arms/B__LOW8.npz |
| secondary: direct per-layer | B | Qwen3-1.7B band zeroed one layer at a time, upstream clean | random band | cut 0.628 vs random max 0.535 | data/Qwen3-1.7B/arms/B__base.npz |
| per-head split | A+B | Qwen3-1.7B zero band, 225 sink heads (calibration s>=0.5) | none | head cut <0.1/0.1-0.5/0.5-0.9/>=0.9 = 73/9/7/136, median 0.99 | data/Qwen3-1.7B/results.json |
| headline | A+B | Qwen3-1.7B pooled | 20 random bands | cut 0.548, random median 0.004, max 0.428, 0/20 >= band; status UNCONFIRMED | data/Qwen3-1.7B/results.json |
| assert | smoke | Qwen3-1.7B | none | gap split err 1.9e-05; bf16 hook loss diff 0.0; k0 identical True | logs/asserts_Qwen3-1.7B.json |
| variant: Qwen3-0.6B | cal | Qwen3-0.6B calibration | none | B={61,62,63}, k=3, share 0.599 | data/Qwen3-0.6B/band.json |
| variant: Qwen3-0.6B | A | Qwen3-0.6B zero band {61,62,63}, all layers | none | cut 0.582 [0.579, 0.585], S 0.499->0.209 | data/Qwen3-0.6B/arms/A__P.npz |
| control: random band | A | Qwen3-0.6B zero [60, 61, 63] | random band (seed 1234) | cut 0.430 | data/Qwen3-0.6B/arms/A__R01.npz |
| control: random band | A | Qwen3-0.6B zero [6, 16, 57] | random band (seed 1234) | cut -0.008 | data/Qwen3-0.6B/arms/A__R02.npz |
| control: random band | A | Qwen3-0.6B zero [7, 33, 50] | random band (seed 1234) | cut -0.012 | data/Qwen3-0.6B/arms/A__R03.npz |
| control: random band | A | Qwen3-0.6B zero [19, 49, 61] | random band (seed 1234) | cut 0.146 | data/Qwen3-0.6B/arms/A__R04.npz |
| control: random band | A | Qwen3-0.6B zero [16, 27, 34] | random band (seed 1234) | cut -0.009 | data/Qwen3-0.6B/arms/A__R05.npz |
| control: random band | A | Qwen3-0.6B zero [40, 53, 55] | random band (seed 1234) | cut 0.030 | data/Qwen3-0.6B/arms/A__R06.npz |
| control: random band | A | Qwen3-0.6B zero [3, 41, 59] | random band (seed 1234) | cut 0.046 | data/Qwen3-0.6B/arms/A__R07.npz |
| control: random band | A | Qwen3-0.6B zero [11, 13, 54] | random band (seed 1234) | cut -0.009 | data/Qwen3-0.6B/arms/A__R08.npz |
| control: random band | A | Qwen3-0.6B zero [3, 10, 29] | random band (seed 1234) | cut -0.002 | data/Qwen3-0.6B/arms/A__R09.npz |
| control: random band | A | Qwen3-0.6B zero [8, 39, 41] | random band (seed 1234) | cut -0.006 | data/Qwen3-0.6B/arms/A__R10.npz |
| control: random band | A | Qwen3-0.6B zero [11, 25, 61] | random band (seed 1234) | cut 0.129 | data/Qwen3-0.6B/arms/A__R11.npz |
| control: random band | A | Qwen3-0.6B zero [0, 12, 33] | random band (seed 1234) | cut 0.004 | data/Qwen3-0.6B/arms/A__R12.npz |
| control: random band | A | Qwen3-0.6B zero [27, 40, 54] | random band (seed 1234) | cut -0.010 | data/Qwen3-0.6B/arms/A__R13.npz |
| control: random band | A | Qwen3-0.6B zero [44, 45, 59] | random band (seed 1234) | cut 0.041 | data/Qwen3-0.6B/arms/A__R14.npz |
| control: random band | A | Qwen3-0.6B zero [22, 49, 62] | random band (seed 1234) | cut 0.223 | data/Qwen3-0.6B/arms/A__R15.npz |
| control: random band | A | Qwen3-0.6B zero [5, 58, 61] | random band (seed 1234) | cut 0.182 | data/Qwen3-0.6B/arms/A__R16.npz |
| control: random band | A | Qwen3-0.6B zero [26, 52, 57] | random band (seed 1234) | cut -0.010 | data/Qwen3-0.6B/arms/A__R17.npz |
| control: random band | A | Qwen3-0.6B zero [53, 54, 62] | random band (seed 1234) | cut 0.213 | data/Qwen3-0.6B/arms/A__R18.npz |
| control: random band | A | Qwen3-0.6B zero [26, 28, 48] | random band (seed 1234) | cut -0.011 | data/Qwen3-0.6B/arms/A__R19.npz |
| control: random band | A | Qwen3-0.6B zero [0, 5, 29] | random band (seed 1234) | cut 0.004 | data/Qwen3-0.6B/arms/A__R20.npz |
| prereg: loss | A | Qwen3-0.6B zero band | median of 20 random bands | dloss +2.686 vs +0.186 nats | data/Qwen3-0.6B/arms/A__P.npz |
| prereg: logit lens | A | Qwen3-0.6B zero band | median of 20 random bands | dlens -0.0448 vs +0.0001 | data/Qwen3-0.6B/arms/A__P.npz |
| secondary: direct per-layer | A | Qwen3-0.6B band zeroed one layer at a time, upstream clean | random band | cut 0.645 vs random max 0.530 | data/Qwen3-0.6B/arms/A__base.npz |
| per-head split | A+B | Qwen3-0.6B zero band, 238 sink heads (calibration s>=0.5) | none | head cut <0.1/0.1-0.5/0.5-0.9/>=0.9 = 68/19/11/140, median 0.99 | data/Qwen3-0.6B/results.json |
| assert | smoke | Qwen3-0.6B | none | gap split err 1.9e-05; bf16 hook loss diff 0.0; k0 identical True | logs/asserts_Qwen3-0.6B.json |
| rebuild: band | cal | Vera own code | none | B={61,62,63} share 0.613, 2000/2000 bootstrap | skeptics/skeptic/vera_band.json |
| headline rebuild | A 0-15 | Vera fp32 own forward | 20 random bands | cut 0.5413 vs bf16 0.5415, max R 0.4243, 0/20 | skeptics/skeptic/vera_fp32_A_0_16.npz |
| law test | A 0-7 | Vera fp32, 30 unseen 3-sets | line from 20 random bands | 29/30 within 0.05, r 0.995 | skeptics/skeptic/vera_pred_law.json |
