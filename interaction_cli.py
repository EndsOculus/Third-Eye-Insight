"""Default CLI shares exactly the desktop v2 boundary; no external AI calls."""
import argparse
import json
from pathlib import Path
from desktop_core import Request, analyze


def run_analysis(argv=None):
    parser = argparse.ArgumentParser(description='Interaction v2: anonymous evidence counts, not intimacy scores')
    parser.add_argument('--db', default='')
    parser.add_argument('--identifier', default='')
    parser.add_argument('--mode', choices=['group','c2c'], default='group')
    parser.add_argument('--start', default='')
    parser.add_argument('--end', default='')
    parser.add_argument('--focus', default='')
    parser.add_argument('--exclude', default='')
    parser.add_argument('--source-format', choices=['nt','tei'], default='nt')
    parser.add_argument('--ai-assist', action='store_true', help='Optional local vector topic candidates, not verified replies')
    parser.add_argument('--encrypted', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--output', help='Optional NEW anonymous JSON file; never overwrites an existing file')
    args = vars(parser.parse_args(argv))
    output = args.pop('output')
    try:
        result = analyze(Request(**args))
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if output:
            with Path(output).open('x',encoding='utf-8') as handle:
                handle.write(text+'\n')
        else:
            print(text)
        return result
    except (ValueError, OSError):
        parser.exit(2, 'Analysis/export failed. Check dates, limits, database schema and a new output path. No private details logged.\n')


if __name__ == '__main__':
    run_analysis()
