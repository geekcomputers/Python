import unittest
import sys
import os

# Add parent directory to path to import backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import BrowserHistory

class TestBrowserHistory(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.browser = BrowserHistory("homepage.com")

    def test_initialization(self):
        """Test proper initialization of BrowserHistory"""
        self.assertEqual(self.browser._curr.val, "homepage.com")
        self.assertEqual(self.browser._back_count, 0)
        self.assertEqual(self.browser._forward_count, 0)
        self.assertIsNone(self.browser._curr.nxt)
        self.assertIsNone(self.browser._curr.prev)

    def test_visit(self):
        """Test visit functionality and forward history cleanup"""
        self.browser.visit("page1.com")
        self.assertEqual(self.browser._curr.val, "page1.com")
        self.assertEqual(self.browser._back_count, 1)
        self.assertEqual(self.browser._forward_count, 0)

        # Test forward history cleanup
        self.browser.visit("page2.com")
        self.browser.back(1)
        self.browser.visit("page3.com")  # Should clear forward history
        self.assertIsNone(self.browser._curr.nxt)
        self.assertEqual(self.browser._forward_count, 0)

    def test_back_navigation(self):
        """Test back navigation with counter validation"""
        # Setup history
        self.browser.visit("page1.com")
        self.browser.visit("page2.com")
        
        # Test normal back navigation
        result = self.browser.back(1)
        self.assertEqual(result, "page1.com")
        self.assertEqual(self.browser._back_count, 1)
        self.assertEqual(self.browser._forward_count, 1)

        # Test back with more steps than available
        result = self.browser.back(5)  # Should only go back 1 step
        self.assertEqual(result, "homepage.com")
        self.assertEqual(self.browser._back_count, 0)
        self.assertEqual(self.browser._forward_count, 2)

    def test_forward_navigation(self):
        """Test forward navigation with counter validation"""
        # Setup history and position
        self.browser.visit("page1.com")
        self.browser.visit("page2.com")
        self.browser.back(2)  # Go back to homepage
        
        # Test normal forward navigation
        result = self.browser.forward(1)
        self.assertEqual(result, "page1.com")
        self.assertEqual(self.browser._forward_count, 1)
        self.assertEqual(self.browser._back_count, 1)

        # Test forward with more steps than available
        result = self.browser.forward(5)  # Should only go forward remaining 1 step
        self.assertEqual(result, "page2.com")
        self.assertEqual(self.browser._forward_count, 0)
        self.assertEqual(self.browser._back_count, 2)

    def test_complex_navigation(self):
        """Test complex navigation patterns"""
        self.browser.visit("page1.com")
        self.browser.visit("page2.com")
        self.browser.visit("page3.com")
        
        # Back navigation
        self.assertEqual(self.browser.back(2), "page1.com")
        
        # New visit should clear forward history
        self.browser.visit("page4.com")
        self.assertEqual(self.browser._forward_count, 0)
        self.assertIsNone(self.browser._curr.nxt)
        
        # Verify we can't go forward to cleared history
        self.assertEqual(self.browser.forward(1), "page4.com")

if __name__ == '__main__':
    unittest.main()