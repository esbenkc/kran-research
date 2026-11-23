# critic2 command at 2026-11-23T10:25:09.426Z (verbatim from the session)
python3 -c 'from fractions import Fraction as F
A_raw=F(60,60); A_bare=F(54,60); A_md=F(60,60); A_anchor=F(60,60)
B_raw=F(31,42); B_md=F(39,42); B_bare=F(3,42); B_anchor=F(41,42)
B_anchor_reach=F(41,42); B_follow=F(39+41,42+41)
print("A-md pred",float(1*A_raw),"err",float(A_md-1*A_raw))
print("A-anchor pred",float(B_anchor_reach*A_raw),"err",float(A_anchor-B_anchor_reach*A_raw))
print("B-raw pred",float(F(1,1)*B_follow),"err",float(B_raw-B_follow))
print("B-bare pred",float(F(1,1)*B_follow),"err",float(B_bare-B_follow))
print("gaps A bare-md",float(A_bare-A_md),"B bare-md",float(F(3,41)-F(39,42)),"B anchor-md",float(F(41,41)-F(39,42)))'
