"""
Produces the lab deliverables from both models:
  generated_sentences.txt   (Part IX / X / XIII generated text)
  normalisation_tests.txt   (Part VII / XII test results)
  cpt_tables.txt            (conditional probability tables)
  results.json              (same data, machine-readable)
"""
import json
import random
from first_order_lm import FirstOrderLM, START, END
from second_order_lm import SecondOrderLM

DATA = """the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug"""
sentences = [line.lower().split() for line in DATA.splitlines()]
train_set = {" ".join(s) for s in sentences}

lm1, lm2 = FirstOrderLM(), SecondOrderLM()
lm1.train(sentences)
lm2.train(sentences)

R = {}

# ---- normalisation tests (sum of every conditional distribution) ----
R["norm1"] = {w: sum(d.values()) for w, d in lm1.probs.items()}
R["norm2"] = {" , ".join(c): sum(d.values()) for c, d in lm2.probs.items()}
with open("normalisation_tests.txt", "w") as f:
    f.write("FIRST-ORDER MODEL: sum over v of P(v | w)\n")
    for w, t in R["norm1"].items():
        f.write(f"  {w:<10} {t:.6f}  {'OK' if abs(t-1)<1e-9 else 'FAIL'}\n")
    f.write("\nSECOND-ORDER MODEL: sum over v of P(v | w1, w2)\n")
    for c, t in R["norm2"].items():
        f.write(f"  ({c})".ljust(26) + f" {t:.6f}  {'OK' if abs(t-1)<1e-9 else 'FAIL'}\n")
    ok = all(abs(t-1) < 1e-9 for t in list(R["norm1"].values()) + list(R["norm2"].values()))
    f.write(f"\nAll distributions sum to 1: {ok}\n")
R["norm_ok"] = ok

# ---- CPTs ----
R["cpt1"] = {w: d for w, d in lm1.probs.items()}
R["cpt2"] = {" , ".join(c): d for c, d in lm2.probs.items()}
with open("cpt_tables.txt", "w") as f:
    f.write("FIRST-ORDER CPT: P(next | current)\n")
    for w, d in lm1.probs.items():
        f.write(f"  {w}: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) + "\n")
    f.write("\nSECOND-ORDER CPT: P(next | previous two)\n")
    for c, d in lm2.probs.items():
        f.write(f"  ({c[0]}, {c[1]}): " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) + "\n")

# ---- generated text ----
random.seed(0)
R["gen1_20"] = [lm1.generate() for _ in range(20)]
random.seed(0)
R["gen2_20"] = [lm2.generate() for _ in range(20)]
random.seed(1)
R["partX_sampling"] = [lm1.generate() for _ in range(5)]
R["partX_greedy"] = [lm1.generate(greedy=True) for _ in range(5)]
R["partX_greedy2"] = [lm2.generate(greedy=True) for _ in range(5)]

# ---- comparison statistics ----
random.seed(2)
g1 = [lm1.generate() for _ in range(1000)]
random.seed(3)
g2 = [lm2.generate() for _ in range(1000)]
R["stats"] = {
    "first": {"contexts": len(lm1.probs), "params": lm1.num_parameters() if hasattr(lm1, "num_parameters") else sum(len(d) for d in lm1.probs.values()),
              "distinct_1000": len(set(g1)), "novel_1000": sum(x not in train_set for x in g1) / 1000,
              "distinct_20": len(set(R["gen1_20"]))},
    "second": {"contexts": len(lm2.probs), "params": lm2.num_parameters(),
               "distinct_1000": len(set(g2)), "novel_1000": sum(x not in train_set for x in g2) / 1000,
               "distinct_20": len(set(R["gen2_20"]))},
}

with open("generated_sentences.txt", "w") as f:
    f.write("PART IX: 20 sampled sentences, first-order model (seed 0)\n")
    for i, s in enumerate(R["gen1_20"], 1): f.write(f"  {i:>2}. {s}\n")
    f.write("\n20 sampled sentences, second-order model (seed 0)\n")
    for i, s in enumerate(R["gen2_20"], 1): f.write(f"  {i:>2}. {s}\n")
    f.write("\nPART X: first-order model, greedy vs sampling (5 each)\n  Greedy (stopped at max_len=30):\n")
    for s in R["partX_greedy"]: f.write(f"    {s}\n")
    f.write("  Sampling (seed 1):\n")
    for s in R["partX_sampling"]: f.write(f"    {s}\n")
    f.write("\nSecond-order model, greedy (5 runs)\n")
    for s in R["partX_greedy2"]: f.write(f"    {s}\n")

json.dump(R, open("results.json", "w"), indent=1)
print(open("normalisation_tests.txt").read()); print(open("generated_sentences.txt").read()); print(R["stats"])
