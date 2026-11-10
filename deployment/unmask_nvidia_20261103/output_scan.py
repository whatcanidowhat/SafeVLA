"""Best-effort size snapshots while episode writers atomically rename outputs.

Only vanished descendants are skipped. Missing roots, access failures and I/O
errors still fail the guard. A subsequent poll sees newly renamed files.
"""
import os
from pathlib import Path
import stat


def _identity(info, path):
    # Windows DirEntry.stat() can report zero inode/device values.
    if info.st_ino:
        return (info.st_dev, info.st_ino)
    return os.path.normcase(os.path.abspath(path))


def used_directory_bytes(root):
    root = Path(root)
    pending = [root]
    seen_dirs = set()
    seen_files = set()
    total = 0
    while pending:
        directory = pending.pop()
        try:
            st = directory.stat()
            key = _identity(st, directory)
            if key in seen_dirs:
                continue
            with os.scandir(directory) as entries:
                seen_dirs.add(key)
                for entry in entries:
                    try:
                        info = entry.stat(follow_symlinks=True)
                        if stat.S_ISDIR(info.st_mode):
                            # Match pathlib.rglob: do not traverse directory symlinks.
                            if not entry.is_symlink():
                                pending.append(Path(entry.path))
                        elif stat.S_ISREG(info.st_mode):
                            key = _identity(info, entry.path)
                            if key not in seen_files:
                                total += info.st_size
                                seen_files.add(key)
                    except FileNotFoundError:
                        # A shard was removed/replaced while finalizing an episode.
                        continue
        except FileNotFoundError:
            if directory == root:
                raise
            # An observed .partial directory became .complete before descent.
            continue
    return total
