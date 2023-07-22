import os
import random
from functools import namedtuple

"""
Target: BlackJack 21 simulate
    - Role
        - Dealer: 1
            - Insurance: (When dealer Get A(1) face up)
                - When dealer got 21
                    - lost chips
                - When dealer doesn't got 21
                    - win double chips (Your Insurance)
        - Player: 1
            - Bet: (Drop chip before gambling start)
            - Hit: (Take other card from the dealer)
            - Stand: (No more card dealer may take card when rank under 17)
            - Double down: (When you got over 10 in first hand)
                           (Get one card)
            - Surrender: (only available as first decision of a hand)
                - Dealer return 50% chips
"""

__author__ = "Alopex Cheung"
__version__ = "0.2"

BLACK_JACK = 21
BASE_VALUE = 17

COLOR = {
    "PURPLE": "\033[1;35;48m",
    "CYAN": "\033[1;36;48m",
    "BOLD": "\033[1;37;48m",
    "BLUE": "\033[1;34;48m",
    "GREEN": "\033[1;32;48m",
    "YELLOW": "\033[1;33;48m",
    "RED": "\033[1;31;48m",
    "BLACK": "\033[1;30;48m",
    "UNDERLINE": "\033[4;37;48m",
    "END": "\033[1;37;0m",
}


class Card:
    __slots__ = "suit", "rank", "is_face"

    def __init__(self, suit, rank, face=True):
        """
        :param suit: patter in the card
        :param rank: point in the card
        :param face: show or cover the face(point & pattern on it)
        """
        self.suit = suit
        self.rank = rank
        self.is_face = face

    def __repr__(self):
        fmt_card = "\t<rank: {rank:2}, suit: {suit:8}>"
        if self.is_face:
            return fmt_card.format(suit=self.suit, rank=self.rank)
        return fmt_card.format(suit="*-Back-*", rank="*-Back-*")

    def show(self):
        print(str(self))


class Deck:
    def __init__(self, num=1):
        """
        :param num: the number of deck
        """
        self.num = num
        self.cards = []
        self.built()

    def __repr__(self):
        return "\n".join([str(card) for card in self.cards])

    def __len__(self):
        return len(self.cards)

    def built(self):
        for _ in range(self.num):
            ranks = [x for x in range(1, 14)]
            suits = "Spades Heart Clubs Diamonds".split()
            for suit in suits:
                for rank in ranks:
                    card = Card(suit, rank)
                    self.cards.append(card)

    def shuffle(self):
        for _ in range(self.num):
            for index in range(len(self.cards)):
                i = random.randint(0, 51)
                self.cards[index], self.cards[i] = self.cards[i], self.cards[index]

    def rebuilt(self):
        self.cards.clear()
        self.built()

    def deliver(self):
        return self.cards.pop()


class Chips:
    def __init__(self, amount):
        """
        :param amount: the chips you own
        """
        self._amount = amount
        self._bet_amount = 0
        self._insurance = 0
        self.is_insurance = False
        self.is_double = False

    def __bool__(self):
        return self.amount > 0

    @staticmethod
    def get_tips(content):
        fmt_tips = "{color}** TIPS: {content}! **{end}"
        return fmt_tips.format(
            color=COLOR.get("YELLOW"), content=content, end=COLOR.get("END")
        )

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, value):
        if not isinstance(value, int):
            type_tips = "Please give a integer"
            raise ValueError(Chips.get_tips(type_tips))
        if value < 0:
            amount_tips = "Your integer should bigger than 0"
            raise ValueError(Chips.get_tips(amount_tips))
        self._amount = value

    @property
    def bet_amount(self):
        return self._bet_amount

    @bet_amount.setter
    def bet_amount(self, value):
        type_tips = "Please give a integer"
        amount_tips = "Your chips should between 1 - " + str(self.amount) + " "
        try:
            value = int(value)
        except ValueError:
            raise ValueError(Chips.get_tips(type_tips))
        else:
            if not isinstance(value, int):
                raise ValueError(Chips.get_tips(type_tips))
            if (value <= 0) or (value > self.amount):
                raise ValueError(Chips.get_tips(amount_tips))
            self._bet_amount = value

    def double_bet(self):
        if self.can_double():
            self._bet_amount *= 2
            self.is_double = True
        else:
            over_tips = "Not enough chips || "
            cannot_double = "CAN'T DO DOUBLE"
            raise ValueError(Chips.get_tips(over_tips + cannot_double))

    @property
    def insurance(self):
        return self._insurance

    @insurance.setter
    def insurance(self, value):
        if self.amount - value < 0:
            over_tips = "Not enough chips"
            raise ValueError(Chips.get_tips(over_tips))
        self._insurance = value
        self.is_insurance = True

    def current_amount(self):
        return self.amount - self.bet_amount - self.insurance

    def reset_chip(self):
        self._bet_amount = 0
        self._insurance = 0
        self.is_double = False
        self.is_insurance = False

    def can_double(self):
        return self.current_amount() - self.bet_amount >= 0


class User:
    def __init__(self, name, role, chips_amount=None, color="END"):
        """
        :param name: User name
        :param role: dealer or player
        :param chips_amount: Casino tokens equal money
        """
        self.name = name
        self.prompt = "{role} >> ({name}) : ".format(role=role, name=self.name)
        self.chips = Chips(chips_amount)
        self.color = color
        self.hand = []
        self.point = 0

    def __repr__(self):
        return str(self.__dict__)

    def obtain_card(self, deck, face=True):
        card = deck.deliver()
        card.is_face = face
        self.hand.append(card)

    def drop_card(self):
        self.hand.clear()
        self.point = 0

    def show_card(self):
        print("\t    ** Here is my card **")
        for card in self.hand:
            card.show()

    def unveil_card(self):
        for card in self.hand:
            card.is_face = True
        self.show_card()

    def calculate_point(self):
        def _extract_rank():
            raw_ranks = [card.rank for card in self.hand]
            cook_ranks = [10 if rank > 10 else rank for rank in raw_ranks]
            return cook_ranks

        def _sum_up(ranks):
            rank_one = sum(ranks)
            rank_eleven = sum([11 if rank == 1 else rank for rank in ranks])
            # Over or has 2 Ace
            if (ranks[::-1] == ranks) and (1 in ranks):
                return 11 + len(ranks) - 1
            if rank_eleven <= BLACK_JACK:
                return rank_eleven
            return rank_one

        points = _extract_rank()
        self.point = _sum_up(points)

    def is_point(self, opt, point):
        self.calculate_point()
        compare_fmt = "{user_point} {opt} {point}".format(
            user_point=self.point, opt=opt, point=point
        )
        return eval(compare_fmt)

    def speak(self, content="", end_char="\n"):
        print("")
        print(
            COLOR.get(self.color) + self.prompt + COLOR.get("END") + content,
            end=end_char,
        )

    def showing(self):
        self.speak()
        self.show_card()

    def unveiling(self):
        self.calculate_point()
        points_fmt = "My point is: {}".format(str(self.point))
        self.speak(points_fmt)
        self.unveil_card()


class Dealer(User):
    def __init__(self, name):
        super().__init__(name=name, role="Dealer", color="PURPLE")
        self.trigger = 0

    def ask_insurance(self):
        buy_insurance = (
            "(Insurance pay 2 to 1)\n"
            "\tMy Face card is an Ace.\n"
            "\tWould your like buy a insurance ?"
        )
        self.speak(content=buy_insurance)

    def strategy_trigger(self, deck):
        if self.is_point("<", BASE_VALUE):
            self.obtain_card(deck)
        else:
            self.trigger += random.randint(0, 5)
            if self.trigger % 5 == 0:
                self.obtain_card(deck)


class Player(User):
    def __init__(self, name, amount):
        super().__init__(name=name, chips_amount=amount, role="Player", color="CYAN")
        self.refresh_prompt()

    def refresh_prompt(self):
        self.prompt = "{role} [ ${remain} ] >> ({name}) : ".format(
            role="Player", name=self.name, remain=self.chips.current_amount()
        )

    def select_choice(self, pattern):
        my_turn = "My turn now."
        self.speak(content=my_turn)
        operation = {
            "I": "Insurance",
            "H": "Hit",
            "S": "Stand",
            "D": "Double-down",
            "U": "Surrender",
        }
        enu_choice = enumerate((operation.get(p) for p in pattern), 1)
        dict_choice = dict(enu_choice)
        for index, operator in dict_choice.items():
            choice_fmt = "\t[{index}] {operation}"
            print(choice_fmt.format(index=index, operation=operator))
        return dict_choice


class Recorder:
    def __init__(self):
        self.data = []
        self.winner = None
        self.remain_chips = 0
        self.rounds = 0
        self.player_win_count = 0
        self.dealer_win_count = 0
        self.player_point = 0
        self.dealer_point = 0

    def update(self, winner, chips, player_point, dealer_point):
        self.rounds += 1
        self.remain_chips = chips
        self.winner = winner
        if self.winner == "Player":
            self.player_win_count += 1
        elif self.winner == "Dealer":
            self.dealer_win_count += 1
        self.player_point = player_point
        self.dealer_point = dealer_point

    def record(self, winner, chips, player_point, dealer_point):
        self.update(winner, chips, player_point, dealer_point)
        Row = namedtuple(
            "Row", ["rounds", "player_point", "dealer_point", "winner", "remain_chips"]
        )
        row = Row(
            self.rounds,
            self.player_point,
            self.dealer_point,
            self.winner,
            self.remain_chips,
        )
        self.data.append(row)

    def draw_diagram(self):
        content = "Record display"
        bars = "--" * 14
        content_bar = bars + content + bars
        base_bar = bars + "-" * len(content) + bars

        os.system("clear")
        print(base_bar)
        print(content_bar)
        print(base_bar)
        self.digram()
        print(base_bar)
        print(content_bar)
        print(base_bar)

    def digram(self):
        title = "Round\tPlayer-Point\tDealer-Point\tWinner-is\tRemain-Chips"
        row_fmt = "{}\t{}\t\t{}\t\t{}\t\t{}"

        print(title)
        for row in self.data:
            print(
                row_fmt.format(
                    row.rounds,
                    row.player_point,
                    row.dealer_point,
                    row.winner,
                    row.remain_chips,
                )
            )

        print("")
        win_rate_fmt = ">> Player win rate: {}%\n>> Dealer win rate: {}%"
        try:
            player_rate = round(self.player_win_count / self.rounds * 100, 2)
            dealer_rate = round(self.dealer_win_count / self.rounds * 100, 2)
        except ZeroDivisionError:
            player_rate = 0
            dealer_rate = 0
        print(win_rate_fmt.format(player_rate, dealer_rate))


class BlackJack:
    def __init__(self, username):
        self.deck = Deck()
        self.dealer = Dealer("Bob")
        self.player = Player(username.title(), 1000)
        self.recorder = Recorder()
        self.go_on = True
        self.first_hand = True
        self.choice = None
        self.winner = None
        self.bust = False
        self.res = None

    def play(self):
        while self.player.chips:
            self.initial_game()
            self.in_bet()
            self.deal_card()
            while self.go_on:
                self.choice = self.menu()
                # self.player.speak()
                self.chips_manage()
                try:
                    self.card_manage()
                except ValueError as res:
                    self.bust = True
                    self.go_on = False
                    self.res = res
            if not self.bust:
                self.is_surrender()
            self.winner = self.get_winner()
            self.res = "Winner is " + self.winner
            os.system("clear")
            self.calculate_chips()
            self.result_exhibit()
            self.dealer.unveiling()
            self.player.unveiling()
            self.recorder.record(
                self.winner,
                self.player.chips.amount,
                self.player.point,
                self.dealer.point,
            )

        self.recorder.draw_diagram()
        ending = "\n\tSorry I lost all chips!\n\tTime to say goodbye."
        self.player.speak(ending)
        print("\n" + "-" * 20 + " End Game " + "-" * 20)

    def initial_game(self):
        self.go_on = True
        self.first_hand = True
        self.choice = None
        self.winner = None
        self.bust = False
        self.deck.rebuilt()
        self.deck.shuffle()
        self.player.chips.reset_chip()
        self.player.drop_card()
        self.player.refresh_prompt()
        self.dealer.drop_card()
        print("\n" + "-" * 20 + " Start Game " + "-" * 20)

    def in_bet(self):
        in_bet = "\n\tI want to bet: "
        not_invalid = True
        self.player.speak(in_bet, end_char="")
        while not_invalid:
            try:
                self.player.chips.bet_amount = input()
            except ValueError as e:
                print(e)
                self.player.speak(in_bet, end_char="")
                continue
            except KeyboardInterrupt:
                print("")
                self.recorder.draw_diagram()
                quit()
            else:
                self.player.refresh_prompt()
                # self.player.speak()
                not_invalid = False

    def deal_card(self):
        # dealer
        self.dealer.obtain_card(self.deck, face=False)
        self.dealer.obtain_card(self.deck)

        # player
        self.player.obtain_card(self.deck)
        self.player.obtain_card(self.deck)

        self.dealer.showing()
        self.player.showing()

    def menu(self):
        pattern = "HS"
        if self.first_hand:
            pattern += "U"
            if self.dealer.hand[1].rank == 1 and self.player.chips.current_amount():
                pattern += "I"
                self.dealer.ask_insurance()
            if self.player.is_point(">", 10) and self.player.chips.can_double():
                pattern += "D"
            self.first_hand = False
        choices = self.player.select_choice(pattern)
        select = self.get_select(len(choices), general_err="Select above number.")
        return choices[select]

    @staticmethod
    def get_select(select_max, prompt=">> ", general_err=""):
        while True:
            try:
                value = input(prompt)
                select = int(value)
                if select > select_max:
                    raise ValueError
            except ValueError:
                print(general_err)
                continue
            except KeyboardInterrupt:
                print("")
                quit()
            else:
                return select

    def chips_manage(self):
        if self.choice == "Insurance":
            err = "The amount should under " + str(self.player.chips.current_amount())
            pay_ins = self.get_select(
                self.player.chips.current_amount(),
                prompt="Insurance amount >> ",
                general_err=err,
            )
            self.player.chips.insurance = pay_ins

        if self.choice == "Double-down":
            try:
                self.player.chips.double_bet()
            except ValueError as e:
                print(e)
        self.player.refresh_prompt()
        if self.choice in ("Insurance", "Double-down", "Surrender"):
            self.go_on = False

    def card_manage(self):
        if self.choice in ("Hit", "Double-down"):
            self.player.obtain_card(self.deck)
            if self.player.is_point(">", BLACK_JACK):
                raise ValueError("Player BUST")
            else:
                self.dealer.strategy_trigger(self.deck)
                if self.dealer.is_point(">", BLACK_JACK):
                    raise ValueError("Dealer BUST")
        elif self.choice != "Surrender":
            if not self.player.chips.is_insurance:
                self.dealer.strategy_trigger(self.deck)
                if self.dealer.is_point(">", BLACK_JACK):
                    raise ValueError("Dealer BUST")

        self.dealer.showing()
        self.player.showing()
        if self.choice in ("Double-down", "Stand"):
            self.go_on = False

    def is_surrender(self):
        if self.choice == "Surrender":
            self.player.speak("Sorry, I surrender....\n")

    def get_winner(self):
        if self.bust:
            return "Dealer" if self.player.is_point(">", BLACK_JACK) else "Player"

        if self.choice == "Surrender":
            return "Dealer"
        elif self.choice == "Insurance":
            if self.player.is_point("==", BLACK_JACK):
                return "Dealer"
            return "Player"

        if self.choice in ("Double-down", "Stand"):
            self.player.calculate_point()
            self.dealer.calculate_point()
            if self.player.point > self.dealer.point:
                return "Player"
            return "Dealer"

        return "Both"

    def calculate_chips(self):
        if self.choice == "Surrender":
            if self.player.chips.bet_amount == 1:
                if self.player.chips.current_amount() == 0:
                    self.player.chips.amount = 0
            else:
                surrender_amount = self.player.chips.bet_amount // 2
                self.player.chips.amount -= surrender_amount

        elif self.choice in ("Double-down", "Stand", "Insurance", "Hit"):
            if self.winner == "Player":
                self.player.chips.amount += (
                    self.player.chips.bet_amount + self.player.chips.insurance * 2
                )
            elif self.winner == "Dealer":
                self.player.chips.amount -= (
                    self.player.chips.bet_amount + self.player.chips.insurance
                )

    def result_exhibit(self):
        def get_color():
            if "BUST" in content:
                return COLOR.get("RED" if "Player" in content else "GREEN")
            if self.winner == "Player":
                return COLOR.get("GREEN")
            elif self.winner == "Dealer":
                return COLOR.get("RED")
            else:
                return COLOR.get("YELLOW")

        end = COLOR.get("END")
        content = str(self.res)
        color = get_color()
        winner_fmt = color + "\n\t>> {content} <<\n" + end
        print(winner_fmt.format(content=content))


def main():
    try:
        user_name = input("What is your name: ")
    except KeyboardInterrupt:
        print("")
    else:
        black_jack = BlackJack(username=user_name)
        black_jack.play()


if __name__ == "__main__":
    main()
