SIGN-OFF: UNCONFIRMED (Positive Control Failed)

I independently recomputed the primary metrics from the OOF scores.
- The base model positive control failed: the megaprobe (arm c) achieved an AUROC of 0.810 on all lies vs truthful, underperforming the linear refit (arm a) at 0.852.
- Due to the predefined rules in PREREG.md, a failure in the positive control downgrades a NO to UNCONFIRMED.
- The mean trained AUROC for spin vs fact-stated truthful on arm c was 0.676 (95% CI: 0.645 - 0.707).
- The mean trained gain of arm c over arm a was -0.057 (95% CI: -0.086 to -0.025).
- Shuffled-label null tests yield AUROCs near 0.5 (c_null: 0.489, a_null: 0.526), ruling out label bias artifacts.
All numbers match the build output.