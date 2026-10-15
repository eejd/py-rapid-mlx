"""The quote helper must select the requested share class and reject bad markup."""

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("goose-finance-quote.py")
spec = importlib.util.spec_from_file_location("goose_finance_quote", SCRIPT)
quote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quote)


class QuoteTests(unittest.TestCase):
    def test_selects_alphabet_quote_after_unrelated_price(self):
        html = (
            '<span jsname="Pdsbrc"><span>$1,073.20</span></span>'
            '<div class="gO24Ff">Alphabet Inc Class A</div>'
            '<div><span jsname="Pdsbrc" class=""><span>$349.08</span></span></div>'
        )
        self.assertEqual(quote.parse_quote(html, "GOOGL"),
                         ("Alphabet Inc Class A", "$349.08"))
        with self.assertRaises(ValueError):
            quote.parse_quote(html, "GOOG")

    def test_missing_quote_fails_closed(self):
        with self.assertRaises(ValueError):
            quote.parse_quote("<html>No current quote</html>", "GOOGL")


if __name__ == "__main__":
    unittest.main()
