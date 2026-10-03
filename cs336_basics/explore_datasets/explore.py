from itertools import islice

file_path = 'C:\\Users\\maric\\OneDrive\\Desktop\\CS336-LLM-from-Scratch\\CS336-LLM-from-Scratch\\data\\TinyStoriesV2-GPT4-train.txt'

# Legge solo le prime 100 righe in streaming
with open(file_path, "r", encoding="utf-8") as f:
    first_100_lines = list(islice(f, 100))

# Stampa a schermo
for i, line in enumerate(first_100_lines, 1):
    print(line.strip())