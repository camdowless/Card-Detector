names = open("names.txt", "r")
cards = list()
for line in names:
    cards.append(line.rstrip())
print(cards)