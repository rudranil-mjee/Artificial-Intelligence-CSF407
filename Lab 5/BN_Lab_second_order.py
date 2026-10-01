"""
Second-order autoregressive language model: P(X_t | X_{t-2}, X_{t-1}).

Plain Python only: dicts, tuples, lists and the `random` module.

What changes compared with the first-order model
  - the context is a PAIR of tokens (a tuple) instead of one token;
  - counts are stored for observed TRIPLES (w1, w2, w3);
  - each row of the conditional table is indexed by a pair (w1, w2);
  - sentences are padded with TWO <START> tokens so that even the first word
    has a full two-token context: P(X1 | <START>, <START>).
"""

import random
from collections import defaultdict

START = "<START>"
END = "<END>"


class SecondOrderLM:
    def __init__(self):
        # counts[(w1, w2)][w3] = C(w1, w2, w3): times w3 followed the pair (w1, w2)
        self.counts = defaultdict(lambda: defaultdict(int))
        # probs[(w1, w2)][w3] = P(w3 | w1, w2)
        self.probs = {}

    # Take tokenised sentences and count triples
    def train(self, sentences):
        self.counts = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            tokens = [START, START] + list(sentence) + [END]
            for w1, w2, w3 in zip(tokens, tokens[1:], tokens[2:]):
                self.counts[(w1, w2)][w3] += 1
        self._build_probabilities()

    # P(w3 | w1, w2) = C(w1, w2, w3) / sum_k C(w1, w2, w_k)
    def _build_probabilities(self):
        self.probs = {}
        for context, nxt_counts in self.counts.items():
            total = sum(nxt_counts.values())
            self.probs[context] = {w: c / total for w, c in nxt_counts.items()}

    # Display the probabilities for a specified pair of previous tokens
    def show(self, w1, w2):
        context = (w1, w2)
        if context not in self.probs:
            print(f"P(next | {w1!r}, {w2!r}): no transitions observed")
            return
        print(f"P(next | {w1!r}, {w2!r}):")
        for w, p in sorted(self.probs[context].items(), key=lambda kv: -kv[1]):
            print(f"  {w:<8} {p:.3f}")

    # Most probable next token (arg max)
    def predict(self, w1, w2):
        context = (w1, w2)
        if context not in self.probs:
            return None  # unseen context: the model has no distribution
        return max(self.probs[context], key=self.probs[context].get)

    # Sample one token from P(. | w1, w2)
    def sample_next(self, w1, w2):
        context = (w1, w2)
        if context not in self.probs:
            return END  # unseen context: end the sentence safely
        words = list(self.probs[context].keys())
        weights = list(self.probs[context].values())
        return random.choices(words, weights=weights, k=1)[0]

    # Generate by repeated sampling (or greedy); stop at <END>
    def generate(self, greedy=False, max_len=30):
        tokens = []
        w1, w2 = START, START
        for _ in range(max_len):
            nxt = self.predict(w1, w2) if greedy else self.sample_next(w1, w2)
            if nxt is None or nxt == END:
                break
            tokens.append(nxt)
            w1, w2 = w2, nxt  # slide the two-token window forward
        return " ".join(tokens)

    # Number of estimated parameters (non-zero table entries)
    def num_parameters(self):
        return sum(len(d) for d in self.probs.values())


def check_normalisation(model, tol=1e-9):
    """Every conditional distribution P(. | w1, w2) must sum to 1."""
    ok = True
    for context, dist in model.probs.items():
        total = sum(dist.values())
        flag = "OK" if abs(total - 1.0) < tol else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"  {str(context):<28} sum = {total:.6f}  {flag}")
    return ok


if __name__ == "__main__":
    data = """the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug"""
    sentences = [line.lower().split() for line in data.splitlines()]

    lm = SecondOrderLM()
    lm.train(sentences)

    print("Selected conditional distributions:\n")
    for ctx in [(START, START), (START, "the"), ("the", "cat"),
                ("on", "the"), ("to", "the"), ("the", "dog")]:
        lm.show(*ctx)
        print(f"  most probable next: {lm.predict(*ctx)}\n")

    print("Unseen context:")
    lm.show("the", "the")
    print()

    print("Normalisation test:")
    assert check_normalisation(lm), "a distribution does not sum to 1"
    print("  All distributions sum to 1.\n")

    # Tests specific to second-order behaviour
    assert abs(lm.probs[("on", "the")]["mat"] - 0.5) < 1e-9
    assert abs(lm.probs[("on", "the")]["rug"] - 0.5) < 1e-9
    assert lm.probs[("to", "the")] == {"park": 1.0}
    assert "park" not in lm.probs[("on", "the")]   # first-order would allow this
    print("Second-order checks passed: the pair (on, the) allows only mat/rug,")
    print("and (to, the) allows only park.\n")

    print(f"Contexts (rows): {len(lm.probs)}   Parameters (non-zero entries): "
          f"{lm.num_parameters()}\n")

    random.seed(0)
    print("Sampled sentences:")
    sampled = [lm.generate() for _ in range(20)]
    for s in sampled:
        print("  ", s)
    print(f"\nDistinct sentences among 20 samples: {len(set(sampled))}")

    print("\nGreedy sentence:", lm.generate(greedy=True))
