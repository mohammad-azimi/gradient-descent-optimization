Gradient Descent report theme update

Extract all files from this update into the existing Gradient Descent project folder, beside run_experiments.py. Replace build_report.py when Windows asks.

The required files are:
build_report.py
polytech_theme.py
polytech_assets/cover_source.png
polytech_assets/header_source.png
polytech_assets/metadata.json

Close the Word report before rebuilding it. In the existing activated Python environment, run:

python validate.py && python build_report.py

The report is written to:
report/Gradient_Descent_Group1_Report.docx

The updated builder reads the current results folder on this computer. It uses those saved experiment values, timings and figures. The theme function checks that all original table contents and the main report text remain identical during formatting.

The visual reference is the supplied Neural_Network_Optimization_Report.docx. Its cover artwork and page banner are reused through native Word crops. Titles are native editable Word text. The cover includes the student names, academic group and instructor supplied in that reference. These fields can be adjusted in polytech_assets/metadata.json.

This update uses the dependencies already listed in the original project requirements.txt.
