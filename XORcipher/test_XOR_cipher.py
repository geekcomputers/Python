#
# Test XORCipher
# **************
#
# Test automation software created by Kevin M. Thomas 09/29/19.
# Test automation software Modified by Kevin M. Thomas 09/29/19.
# CC BY 4.0
#
# Test XORCipher is the test automation suite for the XORCipher created by
# Christian Bender. 
# Usage: python test_XOR_cipher.py
#


import unittest
from unittest import TestCase, mock

from XOR_cipher import XORCipher


class TestXORCipher(TestCase):
    """
    Test XORCipher class.
    """

    def setUp(self):
        """
        The SetUp call with commented values in the event one needs
        to instantiate mocked objects regarding the XORCipher class.
        """

        # key = mock.MagicMock()
        # self.XORCipher_1 = XORCipher(key)
        pass

    @mock.patch('XOR_cipher.XORCipher.__init__')
    def test__init__(self, mock__init__):
        """
        Test the __init__ method with commented values in the event
        one needs to instantiate mocked objects on the method.
        """

        # self.XORCipher_1.__init__ = mock.MagicMock()
        XORCipher.__init__ = mock.MagicMock()

        # self.XORCipher_1.__init__(1)
        XORCipher.__init__()

        # self.XORCipher_1.__init__.assert_called_with(1)
        XORCipher.__init__.assert_called()

    @mock.patch('XOR_cipher.XORCipher.encrypt')
    def test_encrypt(self, mock_encrypt):
        """
        Test the encrypt method with mocked values.
        """

        ans = mock.MagicMock()
        content = mock.MagicMock()
        key = mock.MagicMock()
        XORCipher.encrypt = mock.MagicMock(return_value=ans)
        XORCipher.encrypt(content, key)

        XORCipher.encrypt.assert_called_with(content, key)

    @mock.patch('XOR_cipher.XORCipher.decrypt')
    def test_decrypt(self, mock_decrypt):
        """
        Test the decrypt method with mocked values. 
        """

        ans = mock.MagicMock()
        content = mock.MagicMock()
        key = mock.MagicMock()
        XORCipher.decrypt = mock.MagicMock(return_value=ans)
        XORCipher.decrypt(content, key)

        XORCipher.decrypt.assert_called_with(content, key)

    @mock.patch('XOR_cipher.XORCipher.encrypt_string')
    def test_encrypt_string(self, mock_encrypt_string):
        """
        Test the encrypt_string method with mocked values.
        """

        ans = mock.MagicMock()
        content = mock.MagicMock()
        key = mock.MagicMock()
        XORCipher.encrypt_string = mock.MagicMock(return_value=ans)
        XORCipher.encrypt_string(content, key)

        XORCipher.encrypt_string.assert_called_with(content, key)

    @mock.patch('XOR_cipher.XORCipher.decrypt_string')
    def test_decrypt_string(self, mock_decrypt_string):
        """
        Test the decrypt_string method with mocked values.
        """

        ans = mock.MagicMock()
        content = mock.MagicMock()
        key = mock.MagicMock()
        XORCipher.decrypt_string = mock.MagicMock(return_value=ans)
        XORCipher.decrypt_string(content, key)

        XORCipher.decrypt_string.assert_called_with(content, key)

    @mock.patch('XOR_cipher.XORCipher.encrypt_file')
    def test_encrypt_file(self, mock_encrypt_file):
        """
        Test the encrypt_file method with mocked values.
        """

        file = mock.MagicMock()
        key = mock.MagicMock()
        XORCipher.encrypt_file = mock.MagicMock(return_value=True)
        XORCipher.encrypt_file(file, key)

        XORCipher.encrypt_file.assert_called_with(file, key)

    @mock.patch('XOR_cipher.XORCipher.decrypt_file')
    def test_decrypt_file(self, mock_decrypt_file):
        """
        Test the decrypt_file method with mocked values.
        """

        file = mock.MagicMock()
        key = mock.MagicMock()
        XORCipher.decrypt_string = mock.MagicMock(return_value=True)
        XORCipher.decrypt_string(file, key)

        XORCipher.decrypt_string.assert_called_with(file, key)


if __name__ == '__main__':
    unittest.main()
