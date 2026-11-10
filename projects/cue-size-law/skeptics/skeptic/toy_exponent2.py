# Finer version: eval every step, smaller lr, 6 alphas, local slopes per decade-half.
import numpy as np, json, warnings
warnings.filterwarnings("ignore")
import toy_exponent as T
src = open("toy_exponent.py").read().replace("if t % 5 == 0:", "if True:")
ns = {}; exec(compile(src.replace('if __name__ == "__main__":', 'if False:'), "t", "exec"), ns)
alphas = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
out = {}
for opt, fac, lr in [("sgd", False, 0.1), ("sgd", True, 0.1), ("adam", False, 3e-4), ("adam", True, 3e-4)]:
    key = f"{opt}-{'lora' if fac else 'single'}"
    Ts = [np.median([ns["run"](a, opt, fac, lr, s, Tmax=40000) or np.nan for s in range(3)]) for a in alphas]
    loc = np.diff(np.log(Ts)) / np.diff(np.log(alphas))
    fit = np.polyfit(np.log(alphas), np.log(Ts), 1)[0]
    out[key] = {"alphas": alphas, "T_median": Ts, "local_slopes": loc.round(2).tolist(), "fit_slope": round(fit, 2)}
    print(key, out[key], flush=True)
json.dump(out, open("toy_exponent2.json", "w"), indent=1)
