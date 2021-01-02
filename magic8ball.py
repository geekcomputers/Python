import random

responses = ['It is certain','It is decidedly so','Without a doubt','Yes definitely ','You may rely on it','As I see it, yes','Most likely ','Outlook good','Yes','Signs point to yes','Do not count on it','My reply is no',' My sources say no',' Outlook not so good','Very doubtful', 'Reply hazy try again','Ask again later','Better not tell you now ','Cannot predict now ','Concentrate and ask again']
print("Hi! I am the magic 8 ball, what's your name?")
name = input()
print("Hello!"+ name)

def magic8Ball():
    print("Whay's your question? ")
    question = input()
    answer = responses[random.randint(0,len(responses)-1)]
    print(answer)
    tryAgain()


def tryAgain():
    print("Do you wanna ask any more questions? press Y for yes and any other key to exit ")
    x = input()
    if(x == 'Y'):
        magic8Ball()
    else:
        exit()



magic8Ball()