import os
import unittest
from unittest.mock import patch

from app.config import is_configured


class ConfigurationStatusTests(unittest.TestCase):
    def test_empty_and_placeholder_values_are_not_configured(self) -> None:
        with patch.dict(os.environ, {"OMNIVOICE_TEST_KEY": ""}):
            self.assertFalse(is_configured("OMNIVOICE_TEST_KEY"))

        with patch.dict(os.environ, {"OMNIVOICE_TEST_KEY": "your_key_here"}):
            self.assertFalse(is_configured("OMNIVOICE_TEST_KEY"))

    def test_non_placeholder_value_is_configured(self) -> None:
        with patch.dict(os.environ, {"OMNIVOICE_TEST_KEY": "test-secret-value"}):
            self.assertTrue(is_configured("OMNIVOICE_TEST_KEY"))
