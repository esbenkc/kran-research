# DEVIATIONS — cot-necessity

Every change after the stamped PREREG commit (f2fff5a35887, pushed 2026-11-24T01:10:11Z) gets a line here, with its
commit time. A change inside the FROZEN RULE block starts with "RULE CHANGE:".

- 2026-11-24 02:12 CET, commit ac10a1a53 (Iris, from Bruno #29398; not a rule change): implementation detail of the frozen thought
  definition. grpo.py parse_completion accepts the answer only as ONE final strict tag
  '<answer>\s*(Heads|Tails)\s*</answer>\s*$' (case-insensitive). Text after </answer> (other than whitespace) therefore
  counts as "no answer tag", scored wrong; PREREG.md:61-67 did not say where trailing text goes. The thought = the
  whole completion minus that final tag, with only the literal '<reasoning>'/'</reasoning>' strings removed, as in
  PREREG.
- 2026-11-24 02:11 CET, code f85c69daa (Felix #29402; not a rule change, before any reader data): the reader asks for
  ONE LOWERCASE word and scores logit('heads') - logit('tails') (gpt-oss single tokens 65104 / 54603), with the final
  channel opened right after the prompt. PREREG.md (Reader) said logit(' Heads') - logit(' Tails'), but in gpt-oss's
  tokenizer 'Tails' and ' Tails' are two tokens and 'Heads' one. The calibrated threshold absorbs any word bias. Reader
  prompt texts frozen at reader.py:40-56 in that commit.
- Same commit (Felix #29402; not a rule change): the calibration split has 200 prompts (task.py SPLIT_SIZES, J1 base
  pass), not the 100 in PREREG.md (Reader).
- 2026-11-24 02:25 CET (Bruno #29412 Step 0 data; Felix decision #29415; Tove #29418, Kasper #29414, Vera #29413):
  CONTROL ARM DROPPED, as PREREG.md (Control arm) prescribes when no k qualifies. NDIF Qwen3-8B, n = 400 per k: argmax
  accuracy 0.50 at k = 1, 2, 3, 4 (the model answers Heads on 400/400 at every k), AUROC 1.000 / 0.709 / 0.463 / 0.670;
  no k meets argmax >= 0.85 AND AUROC >= 0.85. Not a rule change; J1 runs the 3 main seeds only. The cause of the
  Heads-every-time answer is not tested (tokenizer 'He' vs 'T'+'ails', or the prompt listing Heads first; Iris #29417).
- Note, not a deviation (Iris): the topic-monitor secondary row asks gpt-oss-20b a yes/no logit question
  ("Is this text reasoning about a coin?", reader.py TOPIC_PROMPT). 2603.30036's monitor (App. D.3.1, "Heads-And-Tails
  Semantic Monitor Prompt") used a judge model with a 0-5 rating and a JSON schema. Same question, different scale
  and model; the post must not call our row "their monitor".
- 2026-11-24 02:39 CET, J1 = labq zfp2nttcl0 at commit 30014f7cc (Bruno #29427, Tove #29421, Vera compute PASS #29428;
  not a rule change, Compute section only): with the control dropped, J1 runs 3 runs, not 5, and the timing rule's
  training budget is 6.0 h, not 7.0 h, so the whole job fits the goal's 8 h: S = min(100, floor((6.0 h - 3 x 6 x
  t_eval) / (3 x t_step))), floored to a multiple of 10 (K = 6 exactly, Kasper #29410); S < 40 -> Qwen3-4B. No
  separate J2: the reader (gate, then score) runs as J1's last stage (Felix #29420).
- 2026-11-24 03:19 CET (Bruno #29483, Vera compute PASS #29484; not a rule change): J1 zfp2nttcl0 sat 39 min unclaimed
  on the laptop queue and was cancelled before it started (no data). Resubmitted as labq 7jau1umj07, same code and
  command plus grpo micro-batch 2 instead of 4 (the same summed gradient per update, about half the backward
  activations) and --mem-gb 16 instead of 24. PREREG fixes 8 prompts x 8 samples per update, not the micro-batch.
- 2026-11-24 04:56 CET (Bruno #29512; not a rule change): J1 7jau1umj07 started 03:49:54Z and was stopped at
  03:54:25Z by the laptop's memory guard (20.4 GB > 16 + 4 GB) during the calibration generation, before any result row
  (only data/logs/calib_k8.log, 2a72faa30). Fix 738d79037: MLX buffer cache bounded (mx.set_cache_limit 1 GB,
  mx.clear_cache() after every generation and update; reader through capped.py). Resubmitted as labq nfpimpkalf at
  commit 738d79037, --mem-gb 20, same command and frozen rule; it started 03:56:34Z.
- 2026-11-24 05:40 CET (Bruno #29516; Kasper #29518, Vera #29519, Iris; not a rule change, the FROZEN block is
  unchanged): J1 nfpimpkalf's frozen timing rule gave S = 20 < 40 on Qwen3-8B, so the job switched to Qwen3-4B-4bit as
  PREREG.md (Model) allows; 8B outputs in data/aborted_Qwen3-8B-4bit/. Clarification, written before any Qwen3-4B
  Step 0 number was posted or committed (commit d4bf44999 at 04:38:31Z; Bruno's file was still uncommitted and
  unread by Iris) and before any 4B eval row: the frozen necessity premise (argmax <= 0.60 AND AUROC <= 0.60 at N = 8) is checked on the model
  that trained. Qwen3-4B's own Step 0 (NDIF, RL format, k = 8, n = 400, 2x2 balanced) must pass the same bar; if it
  fails, the status is UNCONFIRMED (premise failed; 12 or 16 flips are no longer possible, since training used 8). The
  price P was set with a0 = 0.50 from the 8B Step 0; the post states 4B's a0 next to it.
- 2026-11-24 05:40 CET (Bruno #29526, kran-research ee091e39d; follows the line above): Qwen3-4B Step 0 (NDIF, RL
  format, k = 8, n = 400 2x2-balanced; one retry at batch 10 after an NDIF server OOM at batch 50): argmax 0.500, AUROC
  0.506, Heads on 400/400. The premise holds for the trained model; its a0 = 0.50 equals the a0 the job used for P.
