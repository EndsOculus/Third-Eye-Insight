"""Verified explicit local contact schemas only; never infer identities from bodies."""

import json
from pathlib import Path
import unicodedata


def clean_name(value):
    if not isinstance(value, str):
        return ""
    value = "".join(
        c for c in value if not unicodedata.category(c).startswith("C")
    ).strip()
    return value if 0 < len(value) <= 80 else ""


def contact_names(connection, encrypted, users, local_file=None):
    from database_access import DatabaseError

    try:
        return _contact_names(connection, encrypted, users, local_file)
    except DatabaseError as error:
        if error.code == "cancelled":
            raise
        return {}


def _contact_names(connection, encrypted, users, local_file=None):
    users = {str(u) for u in users if str(u).isascii() and str(u).isdigit() and len(str(u))<=32}
    candidates = {}

    def offer(qq, remark, nickname):
        qq = str(qq)
        if qq not in users or not qq.isascii() or not qq.isdigit():
            return
        remark = clean_name(remark)
        nickname = clean_name(nickname)
        if remark or nickname:
            candidates.setdefault(qq, []).append(
                (2 if remark else 1, remark or nickname)
            )

    quote = lambda name: '"' + name.replace('"', '""') + '"'
    tables = connection.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND length(CAST(name AS BLOB))<=64 LIMIT 257"
    )
    if len(tables) <= 256:
        for item in tables:
            table = item["name"]
            # Only explicit named contact tables; numeric NT columns have no assumed semantics.
            if table.lower() not in (
                "contacts",
                "contact",
                "friends",
                "friend",
                "buddy",
                "user_info",
                "contact_info",
                "tei_contacts_v1",
            ):
                continue
            columns = {
                r["name"].lower(): r["name"]
                for r in connection.query(
                    "SELECT name FROM pragma_table_info('"
                    + table.replace("'", "''")
                    + "') WHERE length(CAST(name AS BLOB))<=64 LIMIT 129"
                )
            }
            idcol = next(
                (columns[k] for k in ("qq", "uin", "account_id") if k in columns), None
            )
            remark = next(
                (columns[k] for k in ("remark", "remark_name") if k in columns), None
            )
            nick = next(
                (columns[k] for k in ("nickname", "nick_name", "nick") if k in columns),
                None,
            )
            if not idcol or not (remark or nick):
                continue
            if not users:
                continue
            targets = ",".join("'"+u+"'" for u in users)
            bounded = lambda name, limit: (
                "CASE WHEN length(CAST("
                + quote(name)
                + " AS BLOB))<="
                + str(limit)
                + " THEN "
                + quote(name)
                + " ELSE NULL END"
                if name
                else "NULL"
            )
            sql = (
                "SELECT "
                + bounded(idcol, 32)
                + " AS qq,"
                + bounded(remark, 320)
                + " AS remark,"
                + bounded(nick, 320)
                + " AS nick FROM "
                + quote(table)
                + " WHERE CAST("
                + quote(idcol)
                + " AS TEXT) IN ("
                + targets
                + ") LIMIT 1201"
            )
            found = []
            total_bytes = 0
            for row in connection.iter_query(sql):
                total_bytes += sum(
                    len(str(value).encode("utf-8")) for value in row.values()
                )
                if total_bytes > 1024 * 1024:
                    return {}
                found.append(row)
            if len(found) > 1200:
                continue
            for row in found:
                offer(row["qq"], row["remark"], row["nick"])
    # Explicit documented same-project map, never arbitrary account materials or unrelated files.
    path = (
        Path(local_file)
        if local_file
        else Path(__file__).with_name("contacts.local.json")
    )
    try:
        if path.is_file() and path.stat().st_size <= 256 * 1024:
            data = json.loads(path.read_text("utf-8-sig"))
            if (
                isinstance(data, dict)
                and data.get("schema") == "tei.contacts.v1"
                and isinstance(data.get("contacts"), list)
                and len(data["contacts"]) <= 1000
            ):
                for row in data["contacts"]:
                    if isinstance(row, dict):
                        offer(
                            row.get("qq", ""),
                            row.get("remark", ""),
                            row.get("nickname", ""),
                        )
    except (OSError, ValueError, TypeError):
        pass
    result = {}
    for qq, values in candidates.items():
        priority = max(v[0] for v in values)
        names = {v[1] for v in values if v[0] == priority}
        # Keep an empty conflict marker so weaker native nickname fallback cannot override it.
        result[qq] = names.pop() if len(names) == 1 else ""
    return result
