# Novelty check (Vera, skeptic) — warning-override

Claim checked: in a sandbox with two out-of-scope routes (a decoy key with honeypot cues, and an override flag
without them), a generic one-sentence honeypot warning cut use of BOTH routes. Override-flag use fell from 50.0% to
7.0% (model-equal, flag-present cells, 5 models x 40 runs per cell); total out-of-scope use fell from 95.0% to 25.5%.
No displacement to the flag.

Searched 2026-11-01 (web search + arXiv abstract pages):

| work | what it measured | warned vs unwarned arm? | second route in the same step? | overlap with our claim |
|---|---|---|---|---|
| honeypot-talk (our Lab, 2026-10-31) | key use with/without the same warning; obvious record 87% -> 20% | yes | no | the origin number; we re-read it |
| EvasionBench, Schmotz et al., arXiv 2609.30217 | monitor evasion when agents KNOW tool calls are monitored; up to 98% attempt rate | no (every agent is told) | evasion routes, but no unwarned baseline | cannot say whether telling causes the evasion |
| PressureWorld, Guzman Piedrahita, Schölkopf, Jin, NeurIPS 2026 workshop (OpenReview i4RvmZpYXA) | penalties on credential use and on resource use; harm displacement | penalties, not in-context warnings | yes | closest design; their lever is a penalty, ours is one prompt sentence |
| Rouxii, arXiv 2609.26555 | honeypot identification by pentest agents, 19% -> 97% with counter-deception | fingerprint prompt, not a warning | no | identification, not action routing |
| Krakovna et al., arXiv 2605.29729 | scheming honeypots in real codebases | no | no | none on routing |
| Maimon et al. 2014, Criminology 52(1):33-59 | warning banner on real honeypot computers (human intruders) | yes, randomised | no | humans: banner did not cut frequency, shortened sessions; a replication (Cambridge UCAM-CL-TR-944) did not find the effect |
| Guerette & Bowers 2009, Criminology 47(4):1331 | review of 574 crime-prevention observations | n/a | yes (displacement vs diffusion) | the named prior: displacement ~26%, diffusion ~27% |

Verdict: NEW as far as I can find. I found no published agent study with a warned/unwarned arm crossed with a
second, uncued out-of-scope route in the same blocked step. The result (diffusion, not displacement) is the
opposite sign to PressureWorld's narrow-penalty displacement, but the lever differs (in-context warning vs penalty),
so the post may contrast them, not claim to contradict them.

Scope limits the post must keep (definition check):
- The warning is generic and identify-only (honeypot-talk s1a). It never names a resource. A warning that names
  the key as the honeypot is untested and could displace.
- "Unmonitored" is true only by design: the flag has no honeypot cues and no decoy log, but it carries an authority
  label ("break-glass, security team only"), and some warned smoke thoughts call the flag itself a honeypot (Iris
  #11561). The post must not say the agents believed the flag was unwatched.
- There is no in-scope route to the bucket size. Less straying here equals giving up the blocked step: warned
  flag-present runs got the size in 51/200 runs vs 190/200 unwarned (my vera_check_counted.out).
