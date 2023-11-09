import json
import random
import time
from enum import Enum
from pathlib import Path
from typing import Callable, List

import requests
from colorama import Fore, Style

DEBUG = False
success_code = 200
request_timeout = 1000
data_path = Path(__file__).parent.parent.parent / 'Data'
year = 4800566455


class Source(Enum):
    """Enum that represents switch between local and web word parsing."""

    FROM_FILE = 0  # noqa: WPS115
    FROM_INTERNET = 1  # noqa: WPS115


def print_wrong(text: str, print_function: Callable[[str], None]) -> None:
    """
    Print styled text(red).

    :parameter text: text to print.
    :parameter print_function: Function that will be used to print in game.
    """
    text_to_print = Style.RESET_ALL + Fore.RED + text
    print_function(text_to_print)


def print_right(text: str, print_function: Callable[[str], None]) -> None:
    """
    Print styled text(red).

    :parameter text: text to print.
    :parameter print_function: Function that will be used to print in game.
    """
    print_function(Style.RESET_ALL + Fore.GREEN + text)


def parse_word_from_local(choice_function: Callable[[List[str]], str] = random.choice) -> str:
    # noqa: DAR201
    """
    Parse word from local file.

    :parameter choice_function: Function that will be used to choice a word from file.
    :returns str: string that contains the word.
    :raises FileNotFoundError: file to read words not found.
    """
    try:
        with open(data_path / 'local_words.txt', encoding='utf8') as words_file:
            return choice_function(words_file.read().split('\n'))
    except FileNotFoundError:
        raise FileNotFoundError('File local_words.txt was not found')


def parse_word_from_site(url: str = 'https://random-word-api.herokuapp.com/word') -> str:
    # noqa: DAR201
    """
    Parse word from website.

    :param url: url that word will be parsed from.
    :return Optional[str]: string that contains the word.
    :raises ConnectionError: no connection to the internet.
    :raises RuntimeError: something go wrong with getting the word from site.
    """
    try:
        response: requests.Response = requests.get(url, timeout=request_timeout)
    except ConnectionError:
        raise ConnectionError('There is no connection to the internet')
    if response.status_code == success_code:
        return json.loads(response.content.decode())[0]
    raise RuntimeError('Something go wrong with getting the word from site')


class MainProcess(object):
    """Manages game process."""

    def __init__(self, source: Enum, pr_func: Callable, in_func: Callable, ch_func: Callable) -> None:
        """
        Init MainProcess object.

        :parameter in_func: Function that will be used to get input in game.
        :parameter source: Represents source to get word.
        :parameter pr_func: Function that will be used to print in game.
        :parameter ch_func: Function that will be used to choice word.
        """
        self._source = source
        self._answer_word = ''
        self._word_string_to_show = ''
        self._guess_attempts_coefficient = 2
        self._print_function = pr_func
        self._input_function = in_func
        self._choice_function = ch_func

    def get_word(self) -> str:
        # noqa: DAR201
        """
        Parse word(wrapper for local and web parse).

        :returns str: string that contains the word.
        :raises AttributeError: Not existing enum
        """
        if self._source == Source.FROM_INTERNET:
            return parse_word_from_site()
        elif self._source == Source.FROM_FILE:
            return parse_word_from_local(self._choice_function)
        raise AttributeError('Non existing enum')

    def user_lose(self) -> None:
        """Print text for end of game and exits."""
        print_wrong(f"YOU LOST(the word was '{self._answer_word}')", self._print_function)  # noqa:WPS305

    def user_win(self) -> None:
        """Print text for end of game and exits."""
        print_wrong(f'{self._word_string_to_show} YOU WON', self._print_function)  # noqa:WPS305

    def game_process(self, user_character: str) -> bool:
        # noqa: DAR201
        """
        Process user input.

        :parameter user_character: User character.
        :returns bool: state of game.
        """
        if user_character in self._answer_word:
            word_list_to_show = list(self._word_string_to_show)
            for index, character in enumerate(self._answer_word):
                if character == user_character:
                    word_list_to_show[index] = user_character
            self._word_string_to_show = ''.join(word_list_to_show)
        else:
            print_wrong('There is no such character in word', self._print_function)
        if self._answer_word == self._word_string_to_show:
            self.user_win()
            return True
        return False

    def start_game(self) -> None:
        """Start main process of the game."""
        if time.time() > year:
            print_right('this program is more then 100years age', self._print_function)
        with open(data_path / 'text_images.txt', encoding='utf8') as text_images_file:
            print_wrong(text_images_file.read(), self._print_function)
        print_wrong('Start guessing...', self._print_function)
        self._answer_word = self.get_word()
        self._word_string_to_show = '_' * len(self._answer_word)
        attempts_amount = int(self._guess_attempts_coefficient * len(self._answer_word))
        if DEBUG:
            print_right(self._answer_word, self._print_function)
        for attempts in range(attempts_amount):
            user_remaining_attempts = attempts_amount - attempts
            print_right(f'You have {user_remaining_attempts} more attempts', self._print_function)  # noqa:WPS305
            print_right(f'{self._word_string_to_show} enter character to guess: ', self._print_function)  # noqa:WPS305
            user_character = self._input_function().lower()
            if self.game_process(user_character):
                break
        if '_' in self._word_string_to_show:
            self.user_lose()


if __name__ == '__main__':
    main_process = MainProcess(Source(1), print, input, random.choice)
    main_process.start_game()
