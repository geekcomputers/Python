"""Generates Random Sentences

Creates a sentence by selecting a word at randowm from each of the lists in
    the following order: 'article', 'nounce', 'verb', 'preposition',
    'article' and 'noun'.

    The second part produce a short story consisting of several of
    these sentences -- Random Note Writer!!
"""

import random

article = ["the", "a", "one", "some", "any"]
noun = ["boy", "girl", "dog", "town", "car"]
verb = ["drove", "jumped", "ran", "walked", "skipped"]
preposition = ["to", "from", "over", "under", "on"]


def random_sentence():
    """Creates random and return sentences."""

    sentence = ""
    sentence += article[random.randint(0, 4)] + " " + noun[random.randint(
        0, 4)] + " "
    sentence += verb[random.randint(0, 4)] + " " + preposition[random.randint(
        0, 4)] + " "
    sentence += article[random.randint(0, 4)] + " " + noun[random.randint(
        0, 4)] + ". "
    sentence = sentence[0].upper() + sentence[1:]

    return sentence


# prints random sentences
for x in range(20):
    print(random_sentence())

print()
print()

# creates short story
story = ""
for x in range(20):
    story += random_sentence()

print(story)
