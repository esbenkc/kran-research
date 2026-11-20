# critic2 (openai/gpt-5.5): 2 x 3 Pearson chi-square on the frozen page bins, verbatim from its session.
from scipy.stats import chi2
obs=[1,18,35]; dec=[162,345,174]
G=sum(obs); E=sum(dec)
# 2 x 3 contingency: give-up and continue counts in each bin, expected under common hazard
chi=0
for g,e in zip(obs,dec):
 for o,tot in [(g,e),(e-g,e)]:
  exp=tot*(G/E) if o==g else tot*(1-G/E)
  chi += (o-exp)**2/exp
print(chi, chi2.sf(chi,2))
