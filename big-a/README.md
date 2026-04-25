Data handling
=============

- Purpose: Keep CN data up to date for Qlib-based experiments without real-time streaming.
- Location: data/qlib_data/cn_data/
- Trigger: Run the update_data.py script when you want to pull the latest prebuilt data from chenditc releases.
- How it works:
  - get_last_update_date(data_dir=None): reads calendars/day.txt to return the last update date (YYYY-MM-DD).
  - update_incremental(data_dir=None, start_date=None): downloads the latest qlib_bin tarball from chenditc/investment_data releases and extracts it into the data directory. Performs a light integrity checksum after extraction.
  - verify_update(data_dir=None): basic verification that calendars/day.txt exists and contains a valid date, with sanity checks on instrument lists and sample features.
- CLI usage (dry-run supported):
  - python big-a/scripts/update_data.py --dry-run [--data-dir /path/to/data]
  - python big-a/scripts/update_data.py update --data-dir /path/to/data
- Test a quick status check:
  - In Python: cd big-a && uv run python -c "import sys; sys.path.insert(0,'src'); from big_a.data.updater import get_last_update_date; d = get_last_update_date(); print(f'Last update: {d}')"

Notes:
- Do not attempt real-time updates in Phase 1.
- Do not modify files outside src/big_a/data/updater.py, scripts/update_data.py, and README.md as part of this task.
