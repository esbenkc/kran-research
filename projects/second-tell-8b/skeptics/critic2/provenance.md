# Provenance of the critic2 receipts (staged by Kasper, critic, Claude)

- The scripts here are the exact python bodies critic2 (openai/gpt-5.5) ran inline in its session, extracted from the
  session jsonl by tool-call number (rebuild_k4_v1/v2/final = calls 22-24, null_ratios = 25, subspace_check = 27,
  retention_check = 30). The .out files are the captured tool outputs. critic2 wrote no files itself; the files were
  written after its verdict, so the verdict rests on the inline code they reproduce.
- What critic2 rebuilt vs audited: T_B came from the STORED per-check D_B field of each run JSON (curve[i]['D_B'],
  then its own two-check rule), not from the per-prompt p_off/p_B lists. The ratios, t-intervals and the null ratios
  were recomputed by its own code; null_ratios.py hardcodes the T table printed by rebuild_k4_final.out. The subspace
  numbers were read from data/subspace_k4.json (audited, not recomputed).
- The per-prompt rebuild from p_off/p_B/p_base lists is in Vera's folder (skeptics/skeptic/vera_check.py) and
  Kasper's /tmp/kasper/rebuild.py run (room #31470): 20/20 T_B equal.
- Post changes after the sha critic2 read (athanorscience d95dbe7 -> later heads): k2 paragraph and table, 151-update
  mark, 'halving rule' term, stance line, title. Kasper rechecked every new number against data/k2 and analysis.json
  (room #31584, #31585, #31591): k2 R vs fresh 0.731 [0.658, 0.812], vs each control 0.825 [0.754, 0.903];
  151 = 293.3/1.94. critic2 did not see the k2 numbers.
