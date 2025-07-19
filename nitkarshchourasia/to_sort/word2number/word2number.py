def word_to_number(word: str) -> float:
    # Map number words to their values
    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }

    # Map multipliers to their values
    multipliers = {
        "hundred": 100,
        "thousand": 1000,
        "lakh": 100000,
        "crore": 10000000,
        "billion": 1000000000,
        "trillion": 1000000000000,
    }

    # Split input into words and clean
    words = [
        w.lower()
        for w in word.replace("-", " ").split()
        if w.lower() not in {"and", "point"}
    ]

    total = 0
    current = 0

    for word in words:
        if word in number_words:
            current += number_words[word]
        elif word in multipliers:
            if word == "hundred":
                current *= multipliers[word]
            else:
                # Apply current multiplier and reset
                total += current * multipliers[word]
                current = 0
        else:
            # Handle decimal points or invalid words
            try:
                # Attempt to parse as decimal part
                decimal_part = float(word)
                total += current + decimal_part
                current = 0
            except ValueError:
                raise ValueError(f"Invalid number word: {word}")

    # Add any remaining current value
    total += current
    return total


# Example usage:
number_str = "two trillion seven billion fifty crore thirty-four lakh seven thousand nine hundred"
result = word_to_number(number_str)
print(result)  # Output: 200750347900
