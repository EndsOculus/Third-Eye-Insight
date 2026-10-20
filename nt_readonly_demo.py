"""Local count-only compatibility demonstration; never prints message bodies."""
import argparse
import json
import multiprocessing as mp

from database_access import open_database, DatabaseError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True)
    parser.add_argument('--config', help='Existing local config; no new copy is created')
    args = parser.parse_args()
    try:
        with open_database(args.db, config_file=args.config) as (connection, _, info):
            count = connection.query('SELECT count(*) AS n FROM group_msg_table NOT INDEXED')[0]['n']
            receipt = dict(info, group_rows=int(count), message_bodies_exported=0)
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    except DatabaseError as exc:
        print(json.dumps({'error_code': exc.code, 'message': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
