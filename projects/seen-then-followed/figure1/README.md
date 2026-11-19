# Figure 1 (seen-then-followed)

Built from the counted runs, no model calls.

    python3 pick.py --kmax 27          # example runs: modal (reached, followed) pattern, lowest run index (room #25199, #25958)
    python3 gen.py OUT --kmax 27 --analysis ../results/analysis.json   # writes OUT/figure1.html; asserts K, catch and pred

Head, kicker and subtitle text are in text.json. The blog copy is assets/figures/seen-then-followed-fig1/figure1.html.
