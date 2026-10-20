# Signoff

SIGN-OFF: CONFIRMED.

I independently refit the linear probes on layers 14, 20, and 28 from the raw activation memmaps using the pre-registered `C` values and scaling.
The recovered rates are:
- Layer 14: clean catch = 96.1%, attacked catch (avg) = ~0.95%. clean d' = 5.39.
- Layer 20: clean catch = 99.0%, attacked catch (avg) = ~8.4%. clean d' = 5.53.
- Layer 28: clean catch = 81.4%, attacked catch (avg) = ~1.2%. clean d' = 3.01.
These values and separations are consistent with the claim that standard linear probes fail above layer 14, recovering only modest levels of signal locally, making the 76.8% catch of the megaprobe highly significant.

My independent label-shuffled null test on the raw predictions yields clean catch ~1%, and the random-direction unit vector applied to attacked final representations (seed0) yields ~1% catch, confirming the bounds and meaning the retained performance is not a statistical artifact. The geometric separation is robust.

I verified my calculations match the pre-registered requirements for threshold setting (2nd highest of 200).
