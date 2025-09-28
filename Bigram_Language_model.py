from collections import defaultdict

# Step 1: Training corpus

corpus = [
    "<s> I love NLP </s>",
    "<s> I love deep learning </s>",
    "<s> deep learning is fun </s>"
]

# Tokenize sentences
tokenized_sentences = [sentence.split() for sentence in corpus]

# Step 2: Count unigrams and bigrams

unigram_counts = defaultdict(int)
bigram_counts = defaultdict(int)

for sentence in tokenized_sentences:
    for i, word in enumerate(sentence):
        unigram_counts[word] += 1
        if i < len(sentence) - 1:
            bigram = (word, sentence[i + 1])
            bigram_counts[bigram] += 1

# Step 3: Compute bigram probabilities using MLE
# P(w2 | w1) = Count(w1, w2) / Count(w1)
bigram_probabilities = {}
for (w1, w2), count in bigram_counts.items():
    bigram_probabilities[(w1, w2)] = count / unigram_counts[w1]

# Step 4: Function to calculate probability of a sentence
def sentence_probability(sentence_tokens):
    prob = 1.0
    for i in range(len(sentence_tokens) - 1):
        bigram = (sentence_tokens[i], sentence_tokens[i + 1])
        if bigram in bigram_probabilities:
            prob *= bigram_probabilities[bigram]
        else:
            prob *= 0  # unseen bigram → probability = 0
    return prob

# Step 5: Print counts and probabilities
print("=== Unigram Counts ===")
for word, count in unigram_counts.items():
    print(f"{word}: {count}")

print("\n=== Bigram Counts ===")
for bigram, count in bigram_counts.items():
    print(f"{bigram}: {count}")

print("\n=== Bigram Probabilities (MLE) ===")
for (w1, w2), prob in bigram_probabilities.items():
    print(f"P({w2}|{w1}) = {prob:.4f}")

# Step 6: Test on given sentences
test_sent1 = "<s> I love NLP </s>".split()
test_sent2 = "<s> I love deep learning </s>".split()

prob1 = sentence_probability(test_sent1)
prob2 = sentence_probability(test_sent2)

print("\n=== Sentence Probabilities ===")
print("Sentence 1:", " ".join(test_sent1), "→ Probability:", prob1)
print("Sentence 2:", " ".join(test_sent2), "→ Probability:", prob2)

# Step 7: Compare which one is preferred
if prob1 > prob2:
    print("\nThe model prefers Sentence 1 because it has a higher probability.")
elif prob2 > prob1:
    print("\nThe model prefers Sentence 2 because it has a higher probability.")
else:
    print("\n The model finds both sentences equally likely.")
