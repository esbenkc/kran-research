| headline | s0 | s0@8192 | skeptic rebuild (Vera, own code) | R2 0.4350 | skeptics/skeptic/vera_law_s0at8192_s1at8192.json |
| headline | s1 | s1@8192 | skeptic rebuild (Vera, own code) | R2 0.6547 | skeptics/skeptic/vera_law_s0at8192_s1at8192.json |
| headline | s0,s1 | s0@8192,s1@8192 | skeptic monitor bootstrap (5000) | s0 [0.10, 0.66] P 0.009; s1 [0.45, 0.79] P 0.28 | skeptics/skeptic/vera_robust.json |
| headline | s0,s1 | 20 x-knob settings | skeptic forking check | 0/20 give both seeds R2 >= 0.70 | skeptics/skeptic/vera_robust.json |
| overlap | all | 176 builds | skeptic rebuild of the x axis (own fit) | max abs diff 1.3e-6; classes 91/84/1, 0 mismatches | skeptics/skeptic/vera_x.json |
| control | s0 | shuf-control | skeptic rebuild | max abs drop 0.35 pts | skeptics/skeptic/vera_ctrl.json |
