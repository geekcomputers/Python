from __future__ import print_function
import os
import tweepy

# TODO: Further improvements can be made to the program
# TODO: Further feature improvements and Refactoring can be done to the program
# TODO: Add a README.md file showcasing how adding it to the PATH variable can make the posting much easier


def get_status():
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            break
    return "\n".join(lines)


def tweet_text(api, user):
    print(f"Enter your tweet, {user.name}:")
    tweet = get_status()
    try:
        api.update_status(tweet)
        print("\nTweet posted successfully!")
    except tweepy.TweepError as e:
        print(f"Error posting tweet: {e}")


def tweet_picture(api, user):
    print(f"Enter the picture path, {user.name}:")
    pic = os.path.abspath(input())
    print(f"Enter the status, {user.name}:")
    title = get_status()
    try:
        api.update_with_media(pic, status=title)
        print("\nTweet with picture posted successfully!")
    except tweepy.TweepError as e:
        print(f"Error posting tweet with picture: {e}")


def initialize_api():
    ck = "your_consumer_key"
    cks = "your_consumer_key_secret"
    at = "your_access_token"
    ats = "your_access_token_secret"

    auth = tweepy.OAuthHandler(ck, cks)
    auth.set_access_token(at, ats)
    api = tweepy.API(auth)
    user = api.me()
    return api, user


def main():
    try:
        doit = int(input("\n1. Text\n2. Picture\nChoose option (1/2): "))
        api, user = initialize_api()

        if doit == 1:
            tweet_text(api, user)
        elif doit == 2:
            tweet_picture(api, user)
        else:
            print("Invalid option. Please choose 1 or 2.")
    except ValueError:
        print("Invalid input. Please enter a valid number.")


if __name__ == "__main__":
    main()
