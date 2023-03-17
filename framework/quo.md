[![Downloads](https://pepy.tech/badge/quo)](https://pepy.tech/project/quo)
[![PyPI version](https://badge.fury.io/py/quo.svg)](https://badge.fury.io/py/quo)
[![Wheel](https://img.shields.io/pypi/wheel/quo.svg)](https://pypi.com/project/quo)
[![Windows Build Status](https://img.shields.io/appveyor/build/gerrishons/quo/master?logo=appveyor&cacheSeconds=600)](https://ci.appveyor.com/project/gerrishons/quo)
[![pyimp](https://img.shields.io/pypi/implementation/quo.svg)](https://pypi.com/project/quo)
[![RTD](https://readthedocs.org/projects/quo/badge/)](https://quo.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5848515.svg)](https://doi.org/10.5281/zenodo.5848515)
[![licence](https://img.shields.io/pypi/l/quo.svg)](https://opensource.org/licenses/MIT)
[![Twitter Follow](https://img.shields.io/twitter/follow/gerrishon_s.svg?style=social)](https://twitter.com/gerrishon_s)


[![Logo](https://raw.githubusercontent.com/scalabli/quo/master/pics/quo.png)](https://github.com/scalabli/quo)


`Forever Scalable`

**Quo** is a toolkit for writing Command-Line Interface(CLI) applications and a TUI (Text User Interface) framework for Python.

Quo is making headway towards composing speedy and orderly CLI and TUI applications while forestalling any disappointments brought about by the failure to execute a python application.
Simple to code, easy to learn, and does not come with needless baggage. 

## Compatibility
Quo works flawlessly  with Linux, OSX, and Windows.
Quo requires Python `3.8` or later. 


## Features
- [x] Support for Ansi, RGB and Hex color models
- [x] Support for tabular presentation of data
- [x] Intuitive progressbars
- [x] Code completions
- [x] Nesting of commands
- [x] Customizable Text User Interface _(TUI)_ dialogs.
- [x] Automatic help page generation
- [x] Syntax highlighting
- [x] Autosuggestions
- [x] Key Binders

## Getting Started
### Installation
You can install quo via the Python Package Index (PyPI)

```
pip install -U quo

```

In order to check your installation you can use
```
python -m pip show quo
```
Run the following to test Quo output on your terminal:
```
python -m quo

```
![test](https://github.com/scalabli/quo/raw/master/docs/images/test.png)

:bulb: press ``Ctrl-c`` to exit
# Quo Library
Quo contains a number of builtin features you c
an use to create elegant output in your CLI.

## Quo echo
To output formatted text to your terminal you can import the [echo](https://quo.readthedocs.io/en/latest/introduction.html#quick-start) method.
Try this:

**Example 1**
```python
 from quo import echo

 echo("Hello, World!", fg="red", italic=True, bold=True)
```
<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/print/hello-world.png" />
</p>


**Example 2**
```python
 from quo import echo

 echo("Blue on white", fg="blue", bg="white")
 
```
<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/print/blue-on-white.png" />
</p>


Alternatively, you can import [print](https://quo.readthedocs.io/en/latest/printing_text.html#print)

**Example 1**
```python
 from quo import print

 print('<b>This is bold</b>')
 print('<i>This is italic</i>')
```
**Example 2**

```python
 from quo import print

 print('<u>This is underlined</u>')
 
```
<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/print/underlined1.png" />
</p>

**Example 3**
```python
 from quo import print

 print("Quo is <style bg='red'>Scalable</style>") 
```
<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/print/scalable.png" />
</p>

**Example 4**
```python                   
 # Colors from the ANSI palette.
 print('<red>This is red</red>')
 print('<style fg="white" bg="green">White on green</stlye>')

```
<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/print/white-on-green.png" />
</p>

## Quo prompt
 - Using ``quo.prompt`` method.
```python
 from quo import prompt

 prompt("What is your name?")
```
<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/prompt/prompt1.png" />
</p>


- Using ``quo.prompt.Prompt`` object

**Example 1**

```python
 from quo.prompt import Prompt
   
 session = Prompt()
 session.prompt("Type something:") 
```

**Example 2**

Real time integer validator

```python

 from quo.prompt import Prompt

 session = Prompt(int=True)
 number = int(session.prompt('Give a number: '))

```

![validate](https://raw.githubusercontent.com/scalabli/quo/master/docs/images/number-validator.png)


**Example 3**

``Bottom toolbar``

```python

  from quo.prompt import Prompt
  from quo.text import Text

  def toolbar():
        return Text('This is a <b><style bg="red">Toolbar</style></b>!')

  # Returns a callable
  session = Prompt(bottom_toolbar=toolbar)
  session.prompt('> ')

```

![validate](https://raw.githubusercontent.com/scalabli/quo/master/docs/images/bottom-toolbar.png)


**Example 4**

``Placeholder text``

A placeholder  text that's displayed as long as no input s given.

:bulb: This won't be returned as part of the output.

```python

  from quo.prompt import Prompt
  from quo.text import Text

  session = Prompt(placeholder=Text('<gray>(please type something)</gray>'))
  session.prompt("What is your name?: ")
```
<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/prompt/gray-placeholder.png" />
</p>

**Example 5**

``Coloring the prompt.``


```python

 from quo.color import Color
 from quo.prompt import Prompt

 style = Color("fg:red")
 session = Prompt(style=style)
 session.prompt("Type something: ")

```

<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/prompt/red-prompt.png" />
</p>

**Example 6**

``Autocomplete text``

Press [Tab] to autocomplete
```python

 from quo.prompt import Prompt
 from quo.completion import WordCompleter
 example = WordCompleter(['USA', 'UK', 'Canada', 'Kenya'])
 session = Prompt(completer=example)
 session.prompt('Which country are you from?: ')
```
![Autocompletion](https://github.com/scalabli/quo/raw/master/docs/images/autocompletion.png)

**Example 7**

``Autosuggest text``

Auto suggestion is a way to propose some input completions to the user. Usually, the input is compared to the history and when there is another entry starting with the given text, the completion will be shown as gray text behind the current input.
Pressing the right arrow → or ctrl-e will insert this suggestion, alt-f will insert the first word of the suggestion.
```python

 from quo.history import MemoryHistory
 from quo.prompt import Prompt

 MemoryHistory.append("import os")
 MemoryHistory.append('print("hello")') 
 MemoryHistory.append('print("world")')  
 MemoryHistory.append("import path")

 session = Prompt(history=MemoryHistory, suggest="history")

 while True:
    session.prompt('> ')
```


Read more on [Prompt](https://quo.readthedocs.io/latest/prompt.html)

## Quo Console

For more control over quo terminal content, import and construct a `Console` object.


``Bar``

Draw a horizontal bar with an optional title, which is a good way of dividing your terminal output in to sections.

```python

 from quo.console import Console

 console = Console()
 console.bar("I am a bar")

```

<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/console/bar.png" />
</p>


``Launching Applications``

Quo supports launching applications through `Console.launch`

**Example 1**

```python

 from quo.console import Console

 console = Console()
 console.launch("https://quo.rtfd.io/")

```

**Example 2**

```python

 from quo.console import Console

 console = Console()
 console.launch("/home/path/README.md", locate=True)

```

``Rule``

Used for drawing a horizontal line.

**Example 1**

```python

 from quo.console import Console

 console = Console()
 console.rule()

```
<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/console/rule.png" />
</p>

**Example 2**

A multicolored line.

```python

 from quo.console import Console

 console = Console()
 console.rule(multicolored=True)

```

<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/console/multicolored-rule.png" />
</p>



``Spin``🔁

Quo can create a context manager that is used to display a spinner on stdout as long as the context has not exited

```python

 import time
 from quo.console import Console

 console = Console()

 with console.spin():
           time.sleep(3)
           print("Hello, World")

```
Read more on [Console](https://quo.readthedocs.io/en/latest/console.html)

## Quo Dialogs

High level API for displaying dialog boxes to the user for informational purposes, or to get input from the user.

**Example 1**

Message Box dialog
```python

 from quo.dialog import MessageBox

 MessageBox(
       title="Message pop up window", 
       text="Do you want to continue?\nPress ENTER to quit."
              )

```
![Message Box](https://github.com/scalabli/quo/raw/master/docs/images/messagebox.png)

**Example 2**

Input Box dialog

```python

 from quo.dialog import InputBox

 InputBox(
      title="InputBox shenanigans",
      text="What Country are you from? :"
        )

```
![Prompt Box](https://github.com/scalabli/quo/raw/master/docs/images/promptbox.png)

Read more on [Dialogs](https://quo.readthedocs.io/en/latest/dialogs.html)


## Quo Key Binding🔐

A key binding is an association between a physical key on akeyboard and a parameter.

```python

 from quo import echo
 from quo.keys import bind
 from quo.prompt import Prompt

 session = Prompt()

 # Print "Hello world" when ctrl-h is pressed
 @bind.add("ctrl-h")
 def _(event):
      echo("Hello, World!")

 session.prompt("")

```

Read more on [Key bindings](https://quo.readthedocs.io/en/latest/kb.html)


## Quo Parser 

You can parse optional and positional arguments with Quo and generate help pages for your command-line tools.

```python
 from quo.parse import Parser
 
 parser = Parser(description= "This script prints hello NAME COUNT times.")

 parser.argument('--count', default=3, type=int, help='number of greetings')
 parser.argument('name', help="The person to greet")
 
 arg = parser.parse()
 
 for x in range(arg.count):
     print(f"Hello {arg.name}!")

```

```shell
   $ python prog.py John --count 4
   
```

And what it looks like:

<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/parse/document-scripts.png" />
</p>

Here's what the help page looks like:

```shell
 $ python prog.py --help
```
<p align="center">
  <img src="https://github.com/scalabli/quo/raw/master/docs/images/parse/document-scripts-help.png" />
</p>

Read more on [Parser](https://quo.readthedocs.io/en/latest/parse.html)

## Quo ProgressBar
Creating a new progress bar can be done by calling the class **ProgressBar**
The progress can be displayed for any iterable. This works by wrapping the iterable (like ``range``) with the class **ProgressBar**

```python

 import time
 from quo.progress import ProgressBar
  
 with ProgressBar() as pb:
               for i in pb(range(800)):
                             time.sleep(.01)
```
![ProgressBar](https://raw.githubusercontent.com/scalabli/quo/master/docs/images/simple-progress-bar.png)

Read more on [Progress](https://quo.readthedocs.io/en/latest/progress.html)



## Quo Tables

This offers a number of configuration options to set the look and feel of the table, including how borders are rendered and the style and alignment of the columns.

**Example 1**

```python

 from quo.table import Table

 data = [
     ["Name", "Gender", "Age"],
     ["Alice", "F", 24],
     ["Bob", "M", 19],
     ["Dave", "M", 24]
  ]

 Table(data)

```
![tabulate](https://raw.githubusercontent.com/scalabli/quo/master/docs/images/tables/table.png)

**Example 2**

Right aligned table

```python

 from quo.table import Table

 data = [
    ["Name", "Gender", "Age"],
    ["Alice", "F", 24],
    ["Bob", "M", 19],
    ["Dave", "M", 24]
    ]
 Table(data, align="right")

```

![tabulate](https://raw.githubusercontent.com/scalabli/quo/master/docs/images/tables/right-table.png)

**Example 3**

Colored table

```python

 from quo.table import Table

 data = [
    ["Name", "Gender", "Age"],
    ["Alice", "F", 24],
    ["Bob", "M", 19],
    ["Dave", "M", 24]
    ]
    
 Table(data, style="fg:green")

```


![tabulate](https://raw.githubusercontent.com/scalabli/quo/master/docs/images/tables/colored-table.png)

**Example 4**

Grid table

```python

 from quo.table import Table

 data = [
    ["Name", "Gender", "Age"],
    ["Alice", "F", 24],
    ["Bob", "M", 19],
    ["Dave", "M", 24]
    ]

 Table(data, theme="grid")

```


![tabulate](https://raw.githubusercontent.com/scalabli/quo/master/docs/images/tables/grid-table.png)




Read more on [Table](https://quo.readthedocs.io/en/latest/table.html)

## Quo Widgets
A collection of reusable components for building full screen applications.

``Frame`` 🎞️

Draw a border around any container, optionally with a title.

```python

 from quo import container
 from quo.widget import Frame, Label

 content = Frame(
             Label("Hello, World!"),
               title="Quo: python")

 #Press Ctrl-C to exit
 container(content, bind=True, full_screen=True)

```
![Frame](https://raw.githubusercontent.com/scalabli/quo/master/docs/images/widgets/frame.png)

``Label``

Widget that displays the given text. It is not editable or focusable.

**Example 1**

This will occupy a minimum space in your terminal

```python

 from quo import container
 from quo.widget import Label

 content = Label("Hello, World", style="fg:black bg:red")

 container(content)

```
**Example 2**

This will be a fullscreen application

```python

 from quo import container
 from quo.widget import Label

 content = Label("Hello, World", style="fg:black bg:red")

 # Press Ctrl-C to exit
 container(content, bind=True, full_screen=True)

```
**Example 3**

Full screen application using a custom binding key.

```python

 from quo import container
 from quo.keys import bind
 from quo.widget import Label

 content = Label("Hello, World", style="fg:black bg:red")

 #Press Ctrl-Z to exit
 @bind.add("ctrl-z")
 def _(event):
     event.app.exit()

 container(content, bind=True, full_screen=True)

```

Read more on [Widgets](https://quo.readthedocs.io/en/latest/widgets.html)


For more intricate  examples, have a look in the [examples](https://github.com/scalabli/quo/tree/master/examples) directory and the documentation.

## Donate🎁

In order to for us to maintain this project and grow our community of contributors.
[Donate](https://ko-fi.com/scalabli)



## Quo is...

**Simple**
     If you know Python you can  easily use quo and it can integrate with just about anything.




## Getting Help

### Community

For discussions about the usage, development, and the future of quo, please join our Google community

* [Community👨‍👩‍👦‍👦](https://groups.google.com/g/scalabli)

## Resources

### Bug tracker

If you have any suggestions, bug reports, or annoyances please report them
to our issue tracker at 
[Bug tracker](https://github.com/scalabli/quo/issues/) or send an email to:

 📥 scalabli@googlegroups.com | scalabli@proton.me




## Blogs💻

→ How to build CLIs using [quo](https://www.python-engineer.com/posts/cli-with-quo/)

## License📑

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
This software is licensed under the `MIT License`. See the [License](https://github.com/scalabli/quo/blob/master/LICENSE) file in the top distribution directory for the full license text.


## Code of Conduct
Code of Conduct is adapted from the Contributor Covenant,
version 1.2.0 available at
[Code of Conduct](http://contributor-covenant.org/version/1/2/0/)

