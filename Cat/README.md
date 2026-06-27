# Python Cat

Author: Nitkarsh Chourasia

A small Python implementation of the Unix `cat` command. It reads text from files or standard input and writes the result to standard output.

## Usage

```powershell
python cat.py [options] [files...]
```

If no files are provided, the program reads from standard input.

## Basic Examples

Read one file:

```powershell
python cat.py text_a.txt
```

Read multiple files in order:

```powershell
python cat.py text_a.txt text_b.txt text_c.txt
```

Read from standard input:

```powershell
Get-Content text_a.txt | python cat.py
```

Use `-` to read standard input between files:

```powershell
Get-Content text_b.txt | python cat.py text_a.txt - text_c.txt
```

## Options

Number all lines:

```powershell
python cat.py -n text_a.txt
```

Number only non-empty lines:

```powershell
python cat.py -b text_a.txt
```

Squeeze repeated blank lines:

```powershell
python cat.py -s text_a.txt
```

Show line endings with `$`:

```powershell
python cat.py -E text_a.txt
```

Combine options:

```powershell
python cat.py -n -E text_a.txt
```

```powershell
python cat.py -b -s text_a.txt text_b.txt
```

## Error Handling

If a file cannot be read, the error is printed to standard error and the program continues with the next file.

```powershell
python cat.py text_a.txt missing.txt text_b.txt
```

The program exits with:

- `0` when all input is processed successfully
- `1` when one or more files cannot be read

## Test Cases

Run the automated test suite:

```powershell
python -m unittest test_cat.py
```

The tests cover:

- reading one file
- reading multiple files in order
- reading from standard input
- using `-` for standard input between files
- continuing after a missing file
- `-n` line numbering
- `-b` non-empty line numbering
- `-b` priority over `-n`
- `-s` repeated blank-line squeezing
- `-E` visible line endings
- combined options

## Design Notes

- Files are processed one at a time instead of loading everything into memory.
- `sys.stdin` and opened files are both treated as streams.
- Line numbering continues across multiple files.
- `-b` takes priority over `-n` when both are used.
