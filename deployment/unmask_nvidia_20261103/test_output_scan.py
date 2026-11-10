"""Regression tests use real filesystem renames, no model/simulator calls."""
import errno
from contextlib import ExitStack
import os
import sys
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from output_scan import used_directory_bytes


class OutputScanTests(unittest.TestCase):
    def test_old_failure_and_fixed_scan_during_atomic_episode_rename(self):
        for scanner, expect_failure in [
            (lambda root: sum(p.stat().st_size for p in root.rglob('*') if p.is_file()), True),
            (used_directory_bytes, False),
        ]:
            with self.subTest(old_scanner=expect_failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                partial = root / '0001.partial'
                partial.mkdir()
                (partial / 'metrics.json').write_bytes(b'x' * 51)
                actual_scandir = os.scandir
                renamed = False

                def racing_scandir(path):
                    nonlocal renamed
                    # Glob first visits a directory to match files, then visits
                    # it again for recursive descent. The reported exception is
                    # specifically in _iterate_directories, not wildcard matching.
                    recursive_descent = sys.version_info >= (3, 11)
                    frame = sys._getframe(1)
                    while frame is not None:
                        recursive_descent |= frame.f_code.co_name == '_iterate_directories'
                        frame = frame.f_back
                    if Path(path) == partial and not renamed and (not expect_failure or recursive_descent):
                        partial.rename(root / '0001.complete')
                        renamed = True
                    return actual_scandir(path)

                with ExitStack() as patches:
                    patches.enter_context(patch('os.scandir', racing_scandir))
                    # Python 3.10 captures os.scandir in its pathlib accessor.
                    # Patch that reference as well so the rename occurs in the
                    # original recursive traversal, as in the reported stack.
                    if hasattr(root, '_accessor'):
                        patches.enter_context(patch.object(root._accessor, 'scandir', racing_scandir))
                    elif hasattr(root, '_scandir'):
                        patches.enter_context(patch.object(type(root), '_scandir', racing_scandir))
                    if expect_failure:
                        with self.assertRaises(FileNotFoundError):
                            scanner(root)
                    else:
                        self.assertIn(scanner(root), (0, 51))
                self.assertTrue(renamed)
                self.assertEqual(used_directory_bytes(root), 51)

    def test_removed_file_between_enumeration_and_stat(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gone = root / 'step_000001.npz'
            gone.write_bytes(b'a' * 7)
            (root / 'metrics.json').write_bytes(b'b' * 13)
            actual_scandir = os.scandir

            class FreshEntry:
                def __init__(self, entry):
                    self.path = entry.path

                def stat(self, *, follow_symlinks=True):
                    # Force the uncached stat that Linux performs in this race.
                    return os.stat(self.path, follow_symlinks=follow_symlinks)

                def is_symlink(self):
                    return os.path.islink(self.path)

            class RemovingScan:
                def __enter__(self):
                    with actual_scandir(root) as entries:
                        captured = list(entries)
                    gone.unlink()
                    return iter(FreshEntry(entry) for entry in captured)

                def __exit__(self, *args):
                    return False

            with patch('os.scandir', lambda path: RemovingScan()):
                self.assertEqual(used_directory_bytes(root), 13)

    def test_storage_errors_and_missing_root_are_not_hidden(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for error in [PermissionError(errno.EACCES, 'denied'), OSError(errno.EIO, 'I/O failure')]:
                with self.subTest(error=type(error).__name__), patch('os.scandir', side_effect=error):
                    with self.assertRaises(type(error)):
                        used_directory_bytes(root)
            with self.assertRaises(FileNotFoundError):
                used_directory_bytes(root / 'missing-root')

    def test_real_concurrent_writer_and_repeated_scans(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ready = threading.Event()
            failures = []

            def writer():
                try:
                    for i in range(300):
                        partial = root / f'{i:04d}.partial'
                        partial.mkdir()
                        (partial / 'step.npz').write_bytes(b'shard' * 20)
                        (partial / 'metrics.json').write_bytes(b'm' * 11)
                        ready.set()
                        (partial / 'step.npz').unlink()
                        partial.rename(root / f'{i:04d}.complete')
                except BaseException as exc:
                    failures.append(exc)
                    ready.set()

            thread = threading.Thread(target=writer)
            thread.start()
            ready.wait(5)
            scans = 0
            while thread.is_alive() or scans < 10:
                used_directory_bytes(root)
                scans += 1
            thread.join()
            self.assertFalse(failures)
            self.assertEqual(used_directory_bytes(root), 300 * 11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
