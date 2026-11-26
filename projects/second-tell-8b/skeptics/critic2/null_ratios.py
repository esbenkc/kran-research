# critic2 (openai/gpt-5.5) tool call 25 at 2026-11-26T16:07:59.092Z, cwd /Users/esben/work/theory-1126-oversight-gaming
import math, statistics
T={'fresh':[328,288,296,272,280], 'shuf':[280,288,280,272,328], 'coin':[280,280,272,272,288]}
tcrit=2.7764451051977987
for a in ['coin','shuf']:
    logs=[math.log(T[a][i]/T['fresh'][i]) for i in range(5)]
    mean=sum(logs)/5; sd=statistics.stdev(logs); se=sd/math.sqrt(5)
    print(a,'vs fresh',round(math.exp(mean),6), (round(math.exp(mean-tcrit*se),6), round(math.exp(mean+tcrit*se),6)), [round(math.exp(x),3) for x in logs])
