from __future__ import annotations

import json
import unittest

from memory.records import (
    as_dict_list,
    as_int_list,
    as_text_list,
    parse_json_list,
    parse_text_list,
)


class TextListTests(unittest.TestCase):
    def test_splits_comma_separated_strings(self) -> None:
        self.assertEqual(as_text_list("release, qa ,, "), ["release", "qa"])

    def test_strips_and_keeps_lists(self) -> None:
        self.assertEqual(as_text_list([" release ", "", "qa"]), ["release", "qa"])

    def test_rejects_other_types(self) -> None:
        self.assertEqual(as_text_list(None), [])
        self.assertEqual(as_text_list(3), [])

    def test_parses_json_string_columns(self) -> None:
        self.assertEqual(parse_text_list(json.dumps("a, b")), ["a", "b"])
        self.assertEqual(parse_text_list(json.dumps(["a"])), ["a"])
        self.assertEqual(parse_text_list("[not json"), [])


class JsonListTests(unittest.TestCase):
    def test_string_column_is_split(self) -> None:
        self.assertEqual(parse_json_list(json.dumps("a, b")), ["a", "b"])

    def test_non_list_column_becomes_empty(self) -> None:
        self.assertEqual(parse_json_list(json.dumps({"a": 1})), [])

    def test_dict_list_keeps_only_dicts(self) -> None:
        value = json.dumps([{"from_id": 1}, "nope", 3])

        self.assertEqual(as_dict_list(value), [{"from_id": 1}])

    def test_dict_list_rejects_invalid_json(self) -> None:
        self.assertEqual(as_dict_list("{nope"), [])

    def test_int_list_accepts_json_and_plain_text(self) -> None:
        self.assertEqual(as_int_list("[1, 2]"), [1, 2])
        self.assertEqual(as_int_list("1,2"), [1, 2])
        self.assertEqual(as_int_list([1, "2", "x"]), [1, 2])
        self.assertEqual(as_int_list(None), [])


if __name__ == "__main__":
    unittest.main()
