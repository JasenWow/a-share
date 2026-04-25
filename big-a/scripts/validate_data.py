import sys
from typing import Optional

import typer

from big_a.data.validation import generate_data_report

app = typer.Typer()


def _parse_date(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return str(s)


@app.command()
def main(market: str = "csi300", start_date: Optional[str] = None, end_date: Optional[str] = None) -> None:
    """Validate data quality for the given market and date range."""
    report = generate_data_report(market, _parse_date(start_date), _parse_date(end_date))
    # Pretty-print a compact, readable output
    import json

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
