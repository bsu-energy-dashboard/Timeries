# Chat assistant tools

Timeries can be used through a chat assistant that has access to the Ball State energy dataset. The assistant uses a small set of **MCP tools** behind the scenes. You do not need to know how to call the tools directly; this page explains what is possible so you know what kinds of questions to ask.

MCP tools are best for live questions about meters, dates, usage patterns, unusual spikes, and building contributions. They are designed to return short, useful answers instead of huge spreadsheets.

## What you can ask

Here are examples of natural questions the assistant can answer:

- "Which meters mention Science?"
- "What date range is loaded in the dataset?"
- "Show me a summary of Foundational Science usage in August 2024."
- "Did this meter have suspicious spikes last month?"
- "What were the campus overage events above 22,000 kW?"
- "Which buildings contributed most during the August demand spike?"
- "Does usage increase during the school year?"
- "How does this building's usage change by season?"
- "How does temperature affect this building's energy use?"

## Quick guide

| Tool | What it helps answer | Plain-English example |
| --- | --- | --- |
| `search_meters` | Find the exact meter name for a building or word. | "Find meters for Arts and Journalism." |
| `get_meters` | See what dataset is loaded and get a small sample of meter names. | "How many meters are available?" |
| `get_dataset_info` | Check whether the dataset is ready and what date range it covers. | "What data is loaded right now?" |
| `get_meter_summary` | Get raw statistics for one meter over a date range. | "What was the max demand for SCIENCE-7 last fall?" |
| `get_energy_usage` | Get cleaned usage statistics, and optionally a small sample of hourly values. | "Summarize cleaned usage for this meter in January." |
| `inspect_meter_series` | Look for suspicious sensor spikes before doing analysis. | "Check this meter for weird readings." |
| `analyze_overage` | Find threshold overage events and likely contributing meters. | "What caused the campus demand spikes?" |
| `get_seasonal_profile` | Compare usage by season and session status. | "Is this building higher in summer or winter?" |
| `get_temperature_sensitivity` | Compare usage across outdoor temperature ranges. | "Does usage rise when it is very hot?" |
| `run_trend_tests` | Test whether usage is generally increasing, decreasing, or flat. | "Is this meter trending upward?" |

## Tool details

### Find meters

Use `search_meters` when you know a building name, abbreviation, or word but not the exact meter column.

Good questions:

- "Search for meters with `Science` in the name."
- "Which meters mention `Residence`?"
- "Find anything related to `Foundational`."

The assistant will return matching meter names and usually ask which one you want to analyze.

Use `get_meters` when you only need a broad overview. It returns the meter count, date range, and a small sample. The full meter list can be large, so the assistant usually searches instead of showing everything.

### Check the active dataset

Use `get_dataset_info` to understand what data is loaded.

Good questions:

- "Is the energy dataset ready?"
- "What dates does the current dataset cover?"
- "How many meters are in the dataset?"

This is useful before asking for a specific period. If you request dates outside the loaded range, the assistant can help adjust.

### Summarize one meter

Use `get_meter_summary` for raw statistics from the CSV. "Raw" means the numbers are not cleaned first.

It can answer:

- average usage
- median usage
- minimum and maximum
- total usage
- count of hourly records
- first and third quartiles

Good questions:

- "Give me raw summary stats for `SCIENCE-7` from August 1 to August 31, 2024."
- "What was the maximum value for this meter last semester?"
- "How many hourly readings are available in that range?"

Use this when you want to understand the original data as recorded.

### Summarize cleaned usage

Use `get_energy_usage` when you want a cleaner, analysis-ready summary. This tool applies the default cleaning process before reporting statistics.

Good questions:

- "Summarize cleaned usage for this meter in January 2025."
- "What was the cleaned peak usage?"
- "Can you include a small sample of hourly readings?"

The tool normally returns statistics only. If you ask for hourly values, the assistant can include a small downsampled series, not every row.

### Inspect suspicious spikes

Use `inspect_meter_series` when bad sensor readings may affect results. It looks at the raw meter series and returns the most suspicious high points.

Good questions:

- "Check this meter for suspicious spikes before analysis."
- "Are there any readings that look too high?"
- "Inspect August 2024 for sensor problems."

Important: the assistant should not remove points automatically. It should show you the suspicious timestamps and ask whether you want to remove them. If you approve, those timestamps can be passed as manual outliers in later tools.

### Analyze overage events

Use `analyze_overage` for demand threshold questions. This is the main tool for finding overage events and likely contributors.

It can answer:

- when overage events happened
- how long events lasted
- whether a fixed threshold was exceeded
- what threshold may be reasonable
- which meters contributed most during events
- which buildings were most associated with spikes

Good questions:

- "Find campus overage events above 22,000 kW from April to October 2024."
- "Recommend a demand threshold for campus-wide usage."
- "Which buildings contributed most to the August spike?"
- "Analyze Foundational Science overages using a fixed threshold."

There are two modes:

- `fixed`: use a threshold you provide, such as 22,000 kW.
- `rpca`: estimate a reasonable threshold using the data.

For campus-wide analysis, the raw CSV does not already contain a `totalenergy` column. The assistant must create it from the approved campus total recipe. Likewise, `totalfoundation` is a synthetic total for Foundational Science.

### Seasonal profile

Use `get_seasonal_profile` to understand how a building behaves across seasons and school-session status.

Good questions:

- "How does this building's usage vary by season?"
- "Is usage different when classes are in session?"
- "Compare summer and winter usage for this meter."

The tool returns boxplot-style summary data. Plots are usually omitted in chat to keep the answer readable, but the assistant can include a plot when needed.

### Temperature sensitivity

Use `get_temperature_sensitivity` to compare energy usage across outdoor temperature ranges.

Good questions:

- "Does this building use more energy on hot days?"
- "Compare usage below 45 degrees and above 90 degrees."
- "How temperature-sensitive is this meter?"

This is useful for spotting heating and cooling behavior.

### Trend tests

Use `run_trend_tests` when you want to know whether usage is increasing, decreasing, or staying mostly flat over time.

Good questions:

- "Is this building's usage trending upward?"
- "Run trend tests for this meter over the last year."
- "Is the change statistically meaningful?"

This tool uses trend tests designed for time series data. The assistant should explain the result in plain language, not just show statistical output.

## Special totals

Some useful totals are not raw meter columns. The assistant can create them when needed:

| Name | Meaning |
| --- | --- |
| `totalenergy` | Campus-wide demand total used for campus overage analysis. |
| `totalfoundation` | Foundational Science building total. |

If you ask about campus demand or campus overages, the assistant should use `totalenergy`. If you ask about Foundational Science as a whole, it should use `totalfoundation`.

## Cleaning and outliers

Different tools use data differently:

- `get_meter_summary` uses raw data.
- `inspect_meter_series` inspects raw data.
- `get_energy_usage` returns cleaned usage statistics.
- `analyze_overage` cleans the data by default before analysis.

If suspicious spikes appear, the assistant should:

1. Show you the timestamps and values.
2. Ask whether those points should be removed.
3. Only remove them if you explicitly approve.
4. Re-run the analysis with those approved outliers removed.

This matters because a real energy event and a bad sensor reading can look similar until reviewed.

## Limits

MCP tools are built for conversation, not full data exports.

- Long hourly series are shortened unless you ask for details.
- Overage events and contributors are capped to keep answers readable.
- Plot images are usually omitted in chat.
- Forecasting is not part of the MCP tool set; forecasting is handled through the dashboard or REST API.
- Dataset upload is not part of the MCP tool set; uploads are handled through the dashboard or REST API.

## Good workflow

For most investigations, use this order:

1. Find the right meter with `search_meters`.
2. Check the date range with `get_dataset_info`.
3. Inspect for suspicious spikes with `inspect_meter_series`.
4. Summarize the meter with `get_meter_summary` or `get_energy_usage`.
5. Run the deeper analysis: `analyze_overage`, `get_seasonal_profile`, `get_temperature_sensitivity`, or `run_trend_tests`.

You can ask in ordinary language. The assistant chooses the right tool and explains the result.
