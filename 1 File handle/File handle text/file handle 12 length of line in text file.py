def longlines():
    with open('story.txt', encoding='utf-8') as F:
        lines = F.readlines()

        for i in lines:
            if len(i) < 50:
                print(i, end="\t")


longlines()
