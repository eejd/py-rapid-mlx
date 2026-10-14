---
name: local-documents
description: Read local PDF, CSV and XLSX files and produce checked bookkeeping summaries.
---
Use only files in the selected working folder. Do not use the network, install packages, or start background or detached processes. Finish each command before continuing.
Use `python3` with `pypdf.PdfReader` for PDF, `csv.DictReader` for CSV, and `openpyxl.load_workbook` for XLSX. Never decode PDF bytes by hand.
Use `decimal.Decimal` from strings for every monetary amount. Compute totals per category and separately from all line items; assert they match. Treat wages and withholding as separate facts, not expenses.
Write requested output files in the working folder, with each total label and amount on the same line. Read them back and check every amount. Say when source fields are ambiguous or unreadable. This is bookkeeping, not tax advice or a determination of deductibility.
