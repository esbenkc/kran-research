SIGN-OFF: UNCONFIRMED (Positive Control Failed for all variants)

I independently recomputed the primary metrics from the OOF scores.
- The base model positive control failed for both megaprobe designs against all lies vs truthful: arm c achieved 0.810 and arm c2 (FIBS axial) achieved 0.754, both underperforming the linear refit (arm a) at 0.852.
- Due to the predefined rules in PREREG.md, a failure in the positive control downgrades a NO to UNCONFIRMED.
- The mean trained AUROC for spin vs fact-stated truthful on arm c was 0.676 (95% CI: 0.645 - 0.707).
- The mean trained AUROC for spin vs fact-stated truthful on arm c2 was 0.586.
- The mean trained gain of arm c over arm a was -0.057 (95% CI: -0.086 to -0.025).
- Shuffled-label null tests yield AUROCs near 0.5 (c_null: 0.489, a_null: 0.526, my own label permutations: 0.50), ruling out label bias artifacts.
All numbers match the build output.