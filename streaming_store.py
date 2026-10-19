"""Disk-backed bounded working state; never writes the source database."""

import ctypes
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time

DISK_LIMIT = 8 * 1024**3
MEMORY_LIMIT = 768 * 1024**2
TIME_LIMIT = 15 * 60
BATCH_BYTES = 32 * 1024**2


class ResourceLimit(ValueError):
    pass


def encode(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda x: x.isoformat(),
    )


def decode(value):
    row = json.loads(value)
    row["timestamp"] = datetime.fromisoformat(row["timestamp"])
    return row


class Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
        (name, ctypes.c_size_t)
        for name in (
            "peak",
            "working",
            "qpp",
            "qp",
            "qnp",
            "qn",
            "page",
            "peakpage",
        )
    ]


def resident_bytes():
    if os.name != "nt":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    info = Counters()
    info.cb = ctypes.sizeof(info)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    return (
        info.working
        if query(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb)
        else 0
    )


class DiskSet:
    def __init__(self, store):
        self.store = store
        self.name = store.next_id()

    def add(self, value):
        self.store.db.execute(
            "INSERT OR IGNORE INTO sets VALUES (?,?)", (self.name, encode(value))
        )

    def __contains__(self, value):
        return (
            self.store.db.execute(
                "SELECT 1 FROM sets WHERE name=? AND value=?",
                (self.name, encode(value)),
            ).fetchone()
            is not None
        )

    def __and__(self, other):
        n = self.store.db.execute(
            "SELECT count(*) FROM sets a JOIN sets b ON a.value=b.value WHERE a.name=? AND b.name=?",
            (self.name, other.name),
        ).fetchone()[0]
        return range(n)


class DiskList:
    def __init__(self, store):
        self.store = store
        self.name = store.next_id()
        self.count = 0
        self.total = 0

    def append(self, value):
        self.store.db.execute("INSERT INTO samples VALUES (?,?)", (self.name, value))
        self.count += 1
        self.total += value

    def __len__(self):
        return self.count

    def __iter__(self):
        return (
            r[0]
            for r in self.store.db.execute(
                "SELECT value FROM samples WHERE name=?", (self.name,)
            )
        )

    def median(self):
        offset = (self.count - 1) // 2
        limit = 2 if self.count % 2 == 0 else 1
        values = [
            r[0]
            for r in self.store.db.execute(
                "SELECT value FROM samples WHERE name=? ORDER BY value LIMIT ? OFFSET ?",
                (self.name, limit, offset),
            )
        ]
        return sum(values) / len(values)


class Lookup:
    def __init__(self, store, aliases=False):
        self.store = store
        self.aliases = aliases

    def get(self, key, default=None):
        thread, mid = key
        if mid is None:
            return default
        if self.aliases:
            row = self.store.db.execute(
                "SELECT ord FROM aliases WHERE thread=? AND mid=?", (thread, mid)
            ).fetchone()
            return f"R{row[0] + 1:06d}" if row else default
        row = self.store.db.execute(
            "SELECT record FROM records WHERE thread=? AND mid=? LIMIT 1", (thread, mid)
        ).fetchone()
        return decode(row[0]) if row else default


class Pending:
    def __init__(self, store):
        self.store = store

    def append(self, entry):
        self.store.db.execute(
            "INSERT INTO pending VALUES (?,?,?,?,?,?)",
            (
                entry[2]["timestamp"].isoformat(),
                entry[2].get("thread", entry[2].get("peer", "group")),
                entry[0],
                entry[1],
                entry[2].get("message_id") or "",
                encode(entry),
            ),
        )
        self.store.check()

    def batches(self):
        cursor = self.store.db.execute(
            "SELECT stamp,entry FROM pending ORDER BY stamp,thread,sender,target,mid"
        )
        for stamp, group in itertools.groupby(cursor, key=lambda r: r[0]):
            size = 0
            batch = []
            for _, raw in group:
                size += len(raw.encode("utf-8"))
                if size > BATCH_BYTES:
                    raise ResourceLimit(
                        "A simultaneous event batch exceeds 32 MiB; narrow the selection. No partial result returned."
                    )
                entry = json.loads(raw)
                entry[2]["timestamp"] = datetime.fromisoformat(entry[2]["timestamp"])
                if entry[4] is not None:
                    entry[4]["timestamp"] = datetime.fromisoformat(
                        entry[4]["timestamp"]
                    )
                batch.append(entry)
            yield batch


class RecordStore:
    def __init__(
        self,
        scratch_dir=None,
        excluded=(),
        mode="group",
        cancelled=lambda: False,
        progress=lambda n, t: None,
    ):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="tei-stream-", dir=scratch_dir
        )
        self.path = Path(self.temporary.name) / "working.sqlite"
        self.db = sqlite3.connect(self.path)
        self.db.executescript("""PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-16384; PRAGMA temp_store=MEMORY;
          CREATE TABLE records(seq INTEGER PRIMARY KEY,thread TEXT,mid TEXT,signature TEXT,sender TEXT,peer TEXT,stamp TEXT,record TEXT);
          CREATE INDEX identities ON records(thread,mid);
          CREATE TABLE aliases(thread TEXT,mid TEXT,ord INTEGER,PRIMARY KEY(thread,mid));
          CREATE TABLE sets(name INTEGER,value TEXT,PRIMARY KEY(name,value));
          CREATE TABLE samples(name INTEGER,value REAL); CREATE INDEX sample_values ON samples(name,value);
          CREATE TABLE pending(stamp TEXT,thread TEXT,sender TEXT,target TEXT,mid TEXT,entry TEXT);
        """)
        self.db.execute("PRAGMA max_page_count=" + str(DISK_LIMIT // 4096))
        self.sql_checked = 0
        self.excluded = set(excluded)
        self.mode = mode
        self.cancelled = cancelled
        self.progress = progress
        self.started = time.monotonic()
        self.ticks = 0
        self.serial = 0
        self.raw_rows = 0
        self.excluded_rows = 0
        self.users = set()
        self.native_names = {}
        self.assist = []
        self.count = 0
        self.last_progress = 0
        self.db.set_progress_handler(self.sql_abort, 10000)

    def sql_abort(self):
        if self.cancelled() or time.monotonic() - self.started > TIME_LIMIT:
            return 1
        if time.monotonic() - self.sql_checked < 0.25:
            return 0
        self.sql_checked = time.monotonic()
        try:
            return int(
                resident_bytes() > MEMORY_LIMIT
                or shutil.disk_usage(self.path.parent).free < 64 * 1024**2
                or sum(
                    p.stat().st_size for p in self.path.parent.iterdir() if p.is_file()
                )
                > DISK_LIMIT
            )
        except OSError:
            return 1

    def __enter__(self):
        return self

    def __exit__(self, kind, error, tb):
        self.db.close()
        self.temporary.cleanup()
        if isinstance(error, sqlite3.Error):
            raise ResourceLimit(
                "Working-state query failed or exceeded its time/disk budget. No partial result returned."
            ) from None

    def next_id(self):
        self.serial += 1
        return self.serial

    def new_set(self):
        return DiskSet(self)

    def new_list(self):
        return DiskList(self)

    def check(self, force=False):
        self.ticks += 1
        if not force and self.ticks % 1000:
            return
        if (
            getattr(self, "prepared", False)
            and time.monotonic() - self.last_progress > 0.5
        ):
            self.progress(
                50,
                "Incremental statistics in progress; cancellation remains available.",
            )
            self.last_progress = time.monotonic()
        if self.cancelled():
            raise ResourceLimit("Analysis cancelled; no partial result returned.")
        if time.monotonic() - self.started > TIME_LIMIT:
            raise ResourceLimit(
                "Analysis exceeded its 15-minute budget; narrow the selection. No partial result returned."
            )
        if resident_bytes() > MEMORY_LIMIT:
            raise ResourceLimit(
                "Analysis exceeded its 768 MiB process-memory budget; narrow the selection. No partial result returned."
            )
        if (
            self.path.stat().st_size > DISK_LIMIT
            or shutil.disk_usage(self.path.parent).free < 64 * 1024**2
        ):
            raise ResourceLimit(
                "Working-state disk budget exceeded (8 GiB / 64 MiB reserve); no partial result returned."
            )

    def append(self, row):
        if row["sender"] in self.excluded or (
            self.mode == "c2c" and row["peer"] in self.excluded
        ):
            self.excluded_rows += 1
            return
        row = dict(row, mentions=[t for t in row["mentions"] if t not in self.excluded])
        self.users.add(row["sender"])
        if self.mode == "c2c":
            self.users.add(row["peer"])
        elif row["structure"] == "known":
            self.users.update(row["mentions"])
        if len(self.users) > 300:
            raise ResourceLimit(
                "Selection exceeds the 300-participant pair-state budget; narrow the selection. No partial result returned."
            )
        if getattr(self, 'load_names', False):
            from contact_names import clean_name
            name = clean_name(row.get('local_nickname'))
            if name:
                previous = self.native_names.get(row['sender'])
                if previous is None or row['timestamp'] > previous[0]:
                    self.native_names[row['sender']] = (row['timestamp'],name,False)
                elif row['timestamp'] == previous[0] and name != previous[1]:
                    self.native_names[row['sender']] = (previous[0],previous[1],True)
        self.raw_rows += 1
        if len(self.assist) <= 2000:
            self.assist.append(row)
        thread = row.get("thread", row.get("peer", "group"))
        mid = row.get("message_id")
        signature = encode(
            (
                row["sender"],
                row["timestamp"],
                row["text"],
                row.get("reply_to"),
                row["mentions"],
                row.get("forwarded", False),
                row["structure"],
            )
        )
        self.db.execute(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?)",
            (
                self.raw_rows - 1,
                thread,
                mid,
                signature,
                row["sender"],
                row["peer"],
                row["timestamp"].isoformat(),
                encode(row),
            ),
        )
        if mid:
            self.db.execute(
                "INSERT OR REPLACE INTO aliases VALUES (?,?,?)",
                (thread, mid, row["order"]),
            )
        self.check()

    def prepare(self):
        self.progress(45, "Indexing and checking duplicate/conflicting IDs...")
        self.check(True)
        self.db.execute(
            "CREATE TABLE conflicts AS SELECT thread,mid FROM records WHERE mid IS NOT NULL AND mid != '' GROUP BY thread,mid HAVING min(signature)!=max(signature)"
        )
        self.db.execute("CREATE INDEX conflict_keys ON conflicts(thread,mid)")
        self.conflicts = self.db.execute("SELECT count(*) FROM conflicts").fetchone()[0]
        self.db.execute(
            "CREATE TABLE authors AS SELECT DISTINCT thread,mid,sender FROM records WHERE mid IS NOT NULL"
        )
        self.db.execute("CREATE INDEX author_keys ON authors(thread,mid)")
        from functools import lru_cache

        @lru_cache(maxsize=128)
        def authors(thread, mid):
            return tuple(
                row[0]
                for row in self.db.execute(
                    "SELECT sender FROM authors WHERE thread=? AND mid=?", (thread, mid)
                )
            )

        self.conflict_users = set()
        for raw, thread in self.db.execute(
            "SELECT r.record,r.thread FROM records r JOIN conflicts c ON r.thread=c.thread AND r.mid=c.mid"
        ):
            row = decode(raw)
            self.conflict_users.add(row["sender"])
            if row["structure"] == "known":
                self.conflict_users.update(row["mentions"])
                if row.get("reply_to"):
                    self.conflict_users.update(authors(thread, row["reply_to"]))
        self.db.execute(
            "DELETE FROM records WHERE EXISTS (SELECT 1 FROM conflicts c WHERE c.thread=records.thread AND c.mid=records.mid)"
        )
        clean = self.db.execute("SELECT count(*) FROM records").fetchone()[0]
        self.db.execute(
            "DELETE FROM records WHERE mid IS NOT NULL AND mid != '' AND seq NOT IN (SELECT min(seq) FROM records GROUP BY thread,mid)"
        )
        self.count = self.db.execute("SELECT count(*) FROM records").fetchone()[0]
        self.duplicate_rows = clean - self.count
        self.db.execute("CREATE INDEX chronological ON records(stamp,mid,seq)")
        self.db.execute(
            "CREATE INDEX private_threads ON records(peer,thread,stamp,mid,seq)"
        )
        self.simultaneous = (
            self.count
            - self.db.execute("SELECT count(DISTINCT stamp) FROM records").fetchone()[0]
        )
        self.prepared = True
        self.evidence_keys = Lookup(self, True)
        self.check(True)

    def __len__(self):
        return self.count

    def __iter__(self):
        for (raw,) in self.db.execute(
            "SELECT record FROM records ORDER BY stamp,coalesce(mid,''),seq"
        ):
            self.check()
            yield decode(raw)

    def __getitem__(self, index):
        if index < 0:
            index += self.count
        row = self.db.execute(
            "SELECT record FROM records ORDER BY stamp,coalesce(mid,''),seq LIMIT 1 OFFSET ?",
            (index,),
        ).fetchone()
        if row is None:
            raise IndexError(index)
        return decode(row[0])

    def iter_threads(self):
        cursor = self.db.execute(
            "SELECT peer,thread,record FROM records ORDER BY peer,thread,stamp,coalesce(mid,''),seq"
        )
        for key, group in itertools.groupby(cursor, key=lambda r: (r[0], r[1])):
            yield key, (decode(r[2]) for r in group)

    def bounded_batch(self, rows):
        batch = []
        size = 0
        for row in rows:
            size += len(encode(row).encode("utf-8"))
            if size > BATCH_BYTES:
                raise ResourceLimit(
                    "A simultaneous message batch exceeds 32 MiB; narrow the selection. No partial result returned."
                )
            batch.append(row)
        return batch

    def pending(self):
        return Pending(self)
