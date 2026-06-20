import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CAT_SCRIPT = Path(__file__).with_name("cat.py")


class CatProgramTests(unittest.TestCase):
    def run_cat(self, *args, input_text=""):
        return subprocess.run(
            [sys.executable, str(CAT_SCRIPT), *map(str, args)],
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_reads_single_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "one.txt"
            file_path.write_text("Alpha\nBeta\n", encoding="utf-8")

            result = self.run_cat(file_path)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "Alpha\nBeta\n")
        self.assertEqual(result.stderr, "")

    def test_reads_multiple_files_in_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.txt"
            second = Path(temp_dir) / "second.txt"
            first.write_text("First\n", encoding="utf-8")
            second.write_text("Second\n", encoding="utf-8")

            result = self.run_cat(first, second)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "First\nSecond\n")

    def test_reads_from_stdin_without_files(self):
        result = self.run_cat(input_text="From stdin\nSecond line\n")

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "From stdin\nSecond line\n")

    def test_dash_reads_stdin_between_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.txt"
            second = Path(temp_dir) / "second.txt"
            first.write_text("Before\n", encoding="utf-8")
            second.write_text("After\n", encoding="utf-8")

            result = self.run_cat(first, "-", second, input_text="Middle\n")

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "Before\nMiddle\nAfter\n")

    def test_missing_file_reports_error_and_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.txt"
            missing = Path(temp_dir) / "missing.txt"
            second = Path(temp_dir) / "second.txt"
            first.write_text("Before\n", encoding="utf-8")
            second.write_text("After\n", encoding="utf-8")

            result = self.run_cat(first, missing, second)

        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "Before\nAfter\n")
        self.assertIn("cat:", result.stderr)
        self.assertIn("missing.txt", result.stderr)

    def test_number_all_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "lines.txt"
            file_path.write_text("Alpha\n\nBeta\n", encoding="utf-8")

            result = self.run_cat("-n", file_path)

        expected = "     1\tAlpha\n     2\t\n     3\tBeta\n"
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, expected)

    def test_number_nonblank_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "lines.txt"
            file_path.write_text("Alpha\n\nBeta\n", encoding="utf-8")

            result = self.run_cat("-b", file_path)

        expected = "     1\tAlpha\n\n     2\tBeta\n"
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, expected)

    def test_number_nonblank_takes_priority_over_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "lines.txt"
            file_path.write_text("Alpha\n\nBeta\n", encoding="utf-8")

            result = self.run_cat("-n", "-b", file_path)

        expected = "     1\tAlpha\n\n     2\tBeta\n"
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, expected)

    def test_squeeze_repeated_blank_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "blanks.txt"
            file_path.write_text("Alpha\n\n\nBeta\n\n\n", encoding="utf-8")

            result = self.run_cat("-s", file_path)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "Alpha\n\nBeta\n\n")

    def test_show_ends(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "lines.txt"
            file_path.write_text("Alpha\nBeta", encoding="utf-8")

            result = self.run_cat("-E", file_path)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "Alpha$\nBeta$")

    def test_combines_options(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "lines.txt"
            file_path.write_text("Alpha\n\n\nBeta\n", encoding="utf-8")

            result = self.run_cat("-b", "-s", "-E", file_path)

        expected = "     1\tAlpha$\n$\n     2\tBeta$\n"
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, expected)


if __name__ == "__main__":
    unittest.main()
