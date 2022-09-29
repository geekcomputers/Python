def validate(hand):
    return hand >= 0 and hand <= 2


def print_hand(hand, name="Guest"):
    hands = ["Rock", "Paper", "Scissors"]
    print(f"{name} picked: {hands[hand]}")


def judge(player, computer):
    if player == computer:
        return "Draw"
    elif (
        player == 0
        and computer == 1
        or player == 1
        and computer == 2
        or player == 2
        and computer == 0
    ):
        return "Lose"
    else:
        return "Win"
