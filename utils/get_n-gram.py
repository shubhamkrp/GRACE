import json
import re
import string
import csv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import spacy

# Load spaCy model for tokenization, lemmatization, and NER
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """
    Preprocess the input text by:
      - Converting to lowercase.
      - Removing stopwords, digits, punctuation, and tokens identified as names (PERSON entities).
      - Applying lemmatization.
    Returns a list of processed tokens.
    """
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_stop:
            continue
        if token.like_num or re.search(r'\d', token.text):
            continue
        if token.is_punct or token.text in string.punctuation:
            continue
        if token.ent_type_ == 'PERSON':
            continue
        if not token.is_alpha:
            continue

        # Use the lemma in lowercase
        lemma = token.lemma_.lower()
        tokens.append(lemma)
    return tokens

def extract_trigrams(text):
    """
    Preprocess the text and generate all trigrams (three-word sequences)
    from the list of processed tokens.
    """
    tokens = preprocess_text(text)
    return [' '.join(tokens[i:i+3]) for i in range(len(tokens) - 2)]

# Load JSON files
filenames = ["/home/user/training_data.json"]
nodes = []
for filename in filenames:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
        nodes.extend(data.get('nodes', []))

# Separate collated_notes into IP and OP lists
ip_notes = []
op_notes = []
for node in nodes:
    label = node.get('label', '').strip().upper()
    note = node.get('collated_notes', '')
    if label == 'IP':
        ip_notes.append(note)
    elif label == 'OP':
        op_notes.append(note)

# Generate trigrams and count frequencies separately for IP and OP classes
ip_trigrams = Counter()
for note in ip_notes:
    ip_trigrams.update(extract_trigrams(note))

op_trigrams = Counter()
for note in op_notes:
    op_trigrams.update(extract_trigrams(note))

# Compute combined frequency for each trigram
all_trigrams = set(ip_trigrams.keys()) | set(op_trigrams.keys())
combined_trigrams = {trigram: ip_trigrams.get(trigram, 0) + op_trigrams.get(trigram, 0)
                     for trigram in all_trigrams}

# Save trigram in CSV file sorted by combined frequency (descending)
sorted_trigrams = sorted(combined_trigrams.items(), key=lambda x: x[1], reverse=True)
with open('trigram_frequency_train_only.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Trigram', 'IP_Count', 'OP_Count', 'Combined_Count'])
    for trigram, comb_count in sorted_trigrams:
        ip_count = ip_trigrams.get(trigram, 0)
        op_count = op_trigrams.get(trigram, 0)
        writer.writerow([trigram, ip_count, op_count, comb_count])

ip_specific = {trigram: count for trigram, count in ip_trigrams.items() if trigram not in op_trigrams}
op_specific = {trigram: count for trigram, count in op_trigrams.items() if trigram not in ip_trigrams}

sorted_ip_specific = sorted(ip_specific.items(), key=lambda x: x[1], reverse=True)
sorted_op_specific = sorted(op_specific.items(), key=lambda x: x[1], reverse=True)

# print("Top 5 IP-specific trigrams:")
# for trigram, count in sorted_ip_specific[:5]:
#     print(f"{trigram}: {count}")

# print("\nTop 5 OP-specific trigrams:")
# for trigram, count in sorted_op_specific[:5]:
#     print(f"{trigram}: {count}")

with open('ip_specific_trigrams.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Trigram', 'IP_Count'])
    for trigram, count in sorted_ip_specific:
        writer.writerow([trigram, count])

with open('op_specific_trigrams.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Trigram', 'OP_Count'])
    for trigram, count in sorted_op_specific:
        writer.writerow([trigram, count])





############# Plot are optional #################

top10 = sorted_trigrams[:10]
bottom10 = sorted(sorted_trigrams[-10:], key=lambda x: x[1])
selected_trigrams = top10 + bottom10

trigram_labels = [item[0] for item in selected_trigrams]
ip_counts = [ip_trigrams.get(trigram, 0) for trigram in trigram_labels]
op_counts = [op_trigrams.get(trigram, 0) for trigram in trigram_labels]

x = np.arange(len(trigram_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 8))
rects1 = ax.bar(x - width/2, ip_counts, width, label='IP', color='skyblue')
rects2 = ax.bar(x + width/2, op_counts, width, label='OP', color='lightgreen')

ax.set_xlabel('Trigrams')
ax.set_ylabel('Frequency Count')
ax.set_title('Trigram Frequency Counts for IP and OP Classes (Top 10 and Bottom 10)')
ax.set_xticks(x)
ax.set_xticklabels(trigram_labels, rotation=45, ha='right')
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.tight_layout()
plt.show()

# Plot top 10 IP-specific trigrams
top10_ip_specific = sorted_ip_specific[:15]
ip_specific_labels = [x[0] for x in top10_ip_specific]
ip_specific_values = [x[1] for x in top10_ip_specific]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(ip_specific_labels, ip_specific_values, color='skyblue')
ax.set_xlabel("IP-specific trigrams")
ax.set_ylabel("Frequency Count")
ax.set_title("Top 10 IP-specific Trigrams")
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# top 10 OP specific plot
top10_op_specific = sorted_op_specific[:15]
op_specific_labels = [x[0] for x in top10_op_specific]
op_specific_values = [x[1] for x in top10_op_specific]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(op_specific_labels, op_specific_values, color='lightgreen')
ax.set_xlabel("OP-specific trigrams")
ax.set_ylabel("Frequency Count")
ax.set_title("Top 10 OP-specific Trigrams")
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
