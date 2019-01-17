from textgenrnn import textgenrnn

textgen = textgenrnn()
textgen.train_from_file('montagtext.txt', num_epochs=1)
output = textgen.generate()

print(output)

with open('montagrnntext.txt', 'w') as file:
    file.write(str(output))
