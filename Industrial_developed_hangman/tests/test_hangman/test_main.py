import os
from pathlib import Path
from typing import Callable, List

import pytest
import requests_mock

from src.hangman.main import (
    MainProcess,
    Source,
    parse_word_from_local,
    parse_word_from_site,
)


class FkPrint(object):
    def __init__(self) -> None:
        self.container: List[str] = []

    def __call__(self, value_to_print: str) -> None:
        self.container.append(str(value_to_print))


class FkInput(object):
    def __init__(self, values_to_input: List[str]) -> None:
        self.values_to_input: List[str] = values_to_input

    def __call__(self) -> str:
        return self.values_to_input.pop(0)


@pytest.fixture
def choice_fn() -> Callable:
    return lambda array: array[0]  # noqa: E731


def test_parse_word_from_local() -> None:
    assert isinstance(parse_word_from_local(), str)


def test_parse_word_from_local_error() -> None:
    data_path = Path(os.path.abspath('')) / 'Data'
    real_name = 'local_words.txt'
    time_name = 'local_words_not_exist.txt'

    os.rename(data_path / real_name, data_path / time_name)
    with pytest.raises(FileNotFoundError):
        parse_word_from_local()
    os.rename(data_path / time_name, data_path / real_name)


@pytest.mark.internet_required
def test_parse_word_from_site() -> None:
    assert isinstance(parse_word_from_site(), str)


def test_parse_word_from_site_no_internet() -> None:
    with requests_mock.Mocker() as mock:
        mock.get('https://random-word-api.herokuapp.com/word', text='["some text"]')
        assert parse_word_from_site() == 'some text'


def test_parse_word_from_site_err() -> None:
    with pytest.raises(RuntimeError):
        parse_word_from_site(url='https://www.google.com/dsfsdfds/sdfsdf/sdfds')


def test_get_word(choice_fn: Callable) -> None:
    fk_print = FkPrint()
    fk_input = FkInput(['none'])
    main_process = MainProcess(Source(1), pr_func=fk_print, in_func=fk_input, ch_func=choice_fn)

    assert isinstance(main_process.get_word(), str)


def test_start_game_win(choice_fn: Callable) -> None:
    fk_print = FkPrint()
    fk_input = FkInput(['j', 'a', 'm'])
    main_process = MainProcess(Source(0), pr_func=fk_print, in_func=fk_input, ch_func=choice_fn)

    main_process.start_game()

    assert 'YOU WON' in fk_print.container[-1]


@pytest.mark.parametrize('input_str', [[letter] * 10 for letter in 'qwertyuiopasdfghjklzxcvbnm'])  # noqa: WPS435
def test_start_game_loose(input_str: List[str], choice_fn: Callable) -> None:
    fk_print = FkPrint()
    fk_input = FkInput(input_str)
    main_process = MainProcess(Source(0), pr_func=fk_print, in_func=fk_input, ch_func=choice_fn)

    main_process.start_game()

    assert 'YOU LOST' in fk_print.container[-1]


def test_wow_year(freezer, choice_fn: Callable) -> None:
    freezer.move_to('2135-10-17')
    fk_print = FkPrint()
    fk_input = FkInput(['none'] * 100)  # noqa: WPS435
    main_process = MainProcess(Source(0), pr_func=fk_print, in_func=fk_input, ch_func=choice_fn)

    main_process.start_game()

    assert 'this program' in fk_print.container[0]
