def longlines():
    with open('story.txt', encoding='utf-8') as F:
        line = F.readlines()

        for i in line:
            if len(i) < 50:
                print(i, end="   ")


longlines()
