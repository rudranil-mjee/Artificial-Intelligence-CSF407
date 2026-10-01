"""
First-order autoregressive language model: P(X_t | X_{t-1}).

Plain Python only: dicts, lists and the `random` module.
"""

import random
from collections import defaultdict

START = "<START>"
END = "<END>"


class FirstOrderLM:
    def __init__(self):
        # counts[prev][nxt] = C(prev, nxt), the number of times nxt follows prev
        self.counts = defaultdict(lambda: defaultdict(int))
        # probs[prev][nxt] = P(nxt | prev)
        self.probs = {}

    # 1 + 2. Take tokenised sentences and count transitions
    def train(self, sentences):
        self.counts = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            tokens = [START] + list(sentence) + [END]
            for prev, nxt in zip(tokens, tokens[1:]):
                self.counts[prev][nxt] += 1
        self._build_probabilities()

    # 3. Construct P(X_t | X_{t-1}) = C(w_i, w_j) / sum_k C(w_i, w_k)
    def _build_probabilities(self):
        self.probs = {}
        for prev, nxt_counts in self.counts.items():
            total = sum(nxt_counts.values())
            self.probs[prev] = {w: c / total for w, c in nxt_counts.items()}

    # 4. Display the probabilities for a specified previous token
    def show(self, prev):
        if prev not in self.probs:
            print(f"P(next | {prev!r}): no transitions observed")
            return
        print(f"P(next | {prev!r}):")
        for w, p in sorted(self.probs[prev].items(), key=lambda kv: -kv[1]):
            print(f"  {w:<8} {p:.3f}")

    # 5. Predict the most probable next token (arg max)
    def predict(self, prev):
        if prev not in self.probs:
            return None  # unseen context: the model has no distribution
        return max(self.probs[prev], key=self.probs[prev].get)

    # Sample one token from P(. | prev)
    def sample_next(self, prev):
        if prev not in self.probs:
            return END  # unseen context: end the sentence safely
        words = list(self.probs[prev].keys())
        weights = list(self.probs[prev].values())
        return random.choices(words, weights=weights, k=1)[0]

    # 6 + 7. Generate by repeated sampling; stop at <END>
    def generate(self, greedy=False, max_len=30):
        tokens = []
        prev = START
        for _ in range(max_len):
            nxt = self.predict(prev) if greedy else self.sample_next(prev)
            if nxt is None or nxt == END:
                break
            tokens.append(nxt)
            prev = nxt
        return " ".join(tokens)


if __name__ == "__main__":
    data = """the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug"""
    sentences = [line.lower().split() for line in data.splitlines()]

    lm = FirstOrderLM()
    lm.train(sentences)

    for w in [START, "the", "cat", "sat", "ran"]:
        lm.show(w)
        print(f"  most probable next: {lm.predict(w)}\n")

    # Normalisation check: every row must sum to 1
    for w, dist in lm.probs.items():
        assert abs(sum(dist.values()) - 1.0) < 1e-9, w
    print("All distributions sum to 1.\n")

    random.seed(0)
    print("Sampled sentences:")
    for _ in range(10):
        print("  ", lm.generate())
