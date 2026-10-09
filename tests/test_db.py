import unittest
from unittest import mock

from lol_ticker import db


class SchemaHelperTests(unittest.TestCase):
    def test_setup_preserves_caller_work_and_function_bodies(self):
        conn = mock.MagicMock()
        body = "DO $$ BEGIN PERFORM 'one; two'; PERFORM 2; END; $$;"
        db.apply_schema(conn, (body,))
        self.assertEqual(conn.mock_calls[0], mock.call.commit())
        conn.rollback.assert_not_called()
        conn.execute.assert_any_call(body)

    def test_existing_columns_are_skipped_even_after_inline_comments(self):
        from lol_ticker import wpa, wpx

        for schema in (wpa.SCHEMA, wpx.SCHEMA):
            with self.subTest(schema=schema[0]):
                conn = mock.MagicMock()
                conn.execute.return_value.fetchone.return_value = {"present": 1}
                db.apply_schema(conn, schema)
                executed = [call.args[0] for call in conn.execute.call_args_list]
                self.assertFalse(any(s.lstrip().startswith("ALTER TABLE") for s in executed))
                self.assertTrue(any(s.lstrip().startswith("CREATE TABLE") for s in executed))

    def test_blocked_migration_fails_without_retry_and_exits_transaction(self):
        conn = mock.MagicMock()
        error = db.psycopg.errors.LockNotAvailable("blocked")
        conn.execute.side_effect = [None, error]
        with self.assertRaises(db.psycopg.errors.LockNotAvailable):
            db.apply_schema(conn, ("CREATE TABLE t (a INT)",))
        self.assertEqual(conn.execute.call_count, 2)
        conn.transaction.return_value.__exit__.assert_called_once_with(
            type(error), error, mock.ANY)


if __name__ == "__main__":
    unittest.main()
