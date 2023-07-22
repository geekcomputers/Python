from quo.keys import bind
from quo.prompt import Prompt

session = Prompt()

@bind.add("ctrl-h")
def _(event):
    print("Hello, World")

session.prompt("")
