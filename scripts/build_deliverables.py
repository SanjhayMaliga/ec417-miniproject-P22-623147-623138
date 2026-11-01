"""Build README, six-page report, notebooks and demo content from measured results."""
from pathlib import Path
import json
import sys
import html
import pandas as pd
import nbformat as nbf
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parents[1]

def main():
    summary=pd.read_csv(ROOT/"results/summary.csv").set_index("model")
    audit=json.loads((ROOT/"results/data_audit.json").read_text())
    info=json.loads((ROOT/"results/run_info.json").read_text())
    config=info["config"]
    selection=pd.read_csv(ROOT/"results/validation_selection.csv")
    monthly=pd.read_csv(ROOT/"results/monthly_errors.csv")
    worst=pd.read_csv(ROOT/"results/largest_errors.csv")
    base=float(summary.loc["GRU","mae_mean"])
    improved=float(summary.loc["GRU_calendar","mae_mean"])
    delta=base-improved
    headline=f"Calendar features {'reduced' if delta>0 else 'increased'} GRU 24-hour test MAE by {abs(delta):.3f} degrees C ({abs(delta)/base*100:.1f}%)."
    best=summary.mae_mean.idxmin()
    table="| Model | MAE (C) | RMSE (C) | Train time |\n|---|---:|---:|---:|\n"
    for name,r in summary.iterrows():
        table+=f"| {name} | {r.mae_mean:.3f} +/- {r.mae_std:.3f} | {r.rmse_mean:.3f} +/- {r.rmse_std:.3f} | {r.train_seconds_mean:.1f} s |\n"
    cold=monthly[monthly.model=="GRU_calendar"].sort_values("mae",ascending=False).iloc[0]
    error_text=(f"The calendar GRU's largest monthly mean error occurs in month {int(cold.month)} "
                f"({cold.mae:.3f} C across {int(cold.n)} forecast origins). For seed 17, the largest stored error is "
                f"{worst.iloc[0].absolute_error_calendar:.2f} C at {worst.iloc[0].target_time}: actual "
                f"{worst.iloc[0].actual:.2f} C versus prediction {worst.iloc[0].GRU_calendar:.2f} C. "
                "Rapid temperature changes and missing spatial weather information are possible explanations, not verified causes.")
    methods=(f"Persistence repeats the latest observed temperature. Ridge regression uses all 120 hourly inputs flattened into a vector; "
             f"2015 validation MAE selects alpha={info['ridge_alpha']:g}. The main model is a one-layer GRU with "
             f"{config['hidden_size']} hidden units and four linear outputs (1, 6, 12 and 24 hours). Both GRU variants use these "
             "same outputs, so the sole proposed improvement is adding sine/cosine of hour and day-of-year. "
             f"Adam uses learning rate {config['learning_rate']}, batch size {config['batch_size']}, up to {config['epochs']} epochs, "
             f"patience {config['patience']}, and gradient clipping at 1. Training minimizes mean absolute error across the four "
             "standardized targets. Checkpoints minimize validation MAE at 24 hours. Normalization is reversed before reporting errors.")
    limitations=("This experiment covers one station and one held-out year. Overlapping windows are correlated. Three-seed "
                 "standard deviation quantifies optimization variability, not a confidence interval for future climates. "
                 "Calendar features add a small number of input weights, so parameter counts are not exactly equal. "
                 "Hourly averaging smooths extremes, imputation may distort unusual weather, and timestamps are used as provided without "
                 "assuming a timezone conversion. The selected budget is modest and results do not establish the best possible GRU. "
                 "This educational forecast is unsuitable for safety-critical weather alerts without further validation.")
    ai=("OpenAI Codex assisted with guideline interpretation, implementation, checks, experiments, plots, and draft documentation. "
        "Automated checks and execution are recorded in this project. The students must review the code and results, "
        "record their own verification and contributions, and be able to explain the work at the viva. "
        "This statement does not claim that student review has already occurred.")
    refs=[("MPI weather data and terms","https://www.bgc-jena.mpg.de/wetter/weather_data.html"),
          ("TensorFlow time-series tutorial and archive provenance","https://www.tensorflow.org/tutorials/structured_data/time_series"),
          ("Cho et al. (2014), Learning Phrase Representations using RNN Encoder-Decoder","https://arxiv.org/abs/1406.1078"),
          ("Kingma and Ba (2014), Adam","https://arxiv.org/abs/1412.6980"),
          ("scikit-learn Ridge documentation","https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html")]
    readme=f'''# JenaClimateForecasting

**EC417 P22.** {headline}

**M Sanjhay Baargav (623147)** and **Trinesh Kumar (623138)**  
Department of Electronics and Communication Engineering, NIT Andhra Pradesh.  
Repository name: `ec417-miniproject-P22-623147-623138`.

## Problem

Predict air temperature 24 hours ahead from the previous five days of hourly weather observations. Short-range temperature forecasting supports planning and provides a clear test of temporal modelling. Primary metric: mean absolute error (MAE) in degrees Celsius, because it directly expresses typical absolute error. Secondary metric: RMSE, which penalizes large mistakes more strongly.

## Data

Jena Climate 2009-2016: {audit['raw_rows']:,} raw records and 14 original variables. Source, CC BY 4.0 terms, download URL, attribution, hashes and preprocessing details are in [data/README.md](data/README.md) and [source_manifest.json](data/source_manifest.json). Class balance is not applicable to regression. Raw timestamps split into 2009-2014 training, 2015 validation and 2016 test before hourly resampling and window creation. Forward-fill within each period, then training medians. Training-only normalization. Invalid sentinel counts and exact boundaries are in [data_audit.json](results/data_audit.json).

All learned models use {audit['windows']['train']:,} training windows at stride {config['train_stride']}; validation and test use hourly origins ({audit['windows']['validation']:,} and {audit['windows']['test']:,} windows). No input window or target crosses a split. Targets without observed hourly temperature are excluded from evaluation.

## Method

{methods}

```text
Raw time split -> hourly means -> train-only preprocessing -> 120-hour windows
                                                      -> persistence / ridge / GRU
                                                      -> GRU + four calendar inputs
```

## Results

Test results at 24 hours. Mean +/- sample standard deviation across seeds 17, 42 and 147 for GRUs. Persistence and ridge are deterministic, evaluated once; their zero seed spread does not mean zero sampling uncertainty. Ridge time includes its validation penalty search; GRU time includes validation and early stopping, averaged per seed. Download/preprocessing/report time is excluded.

{table}

{headline} Lowest observed mean MAE: **{best}**. Test results were computed after all model selection finished; no test score selected an epoch or penalty. This comparison does not claim statistical significance.

![MAE versus horizon](results/figures/horizon_mae.png)
![Two test weeks](results/figures/two_test_weeks.png)
![Validation learning curves](results/figures/learning_curves.png)

## Error analysis

{error_text}

The two-week chart uses the first eligible test fortnight and seed 17, chosen before inspecting results. Monthly errors average absolute errors across the three seeds, not ensemble predictions. See [monthly errors](results/monthly_errors.csv), [largest errors](results/largest_errors.csv), and the additional figures in `results/figures/`.

## Limitations and ethics

{limitations}

## Reproduce

Python 3.12 was used. From this directory, create an environment and run:

```bash
python -m venv .venv
# Windows PowerShell:
.venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_project.py
python -m unittest discover -s tests -v
```

One command downloads the archived data, trains all models, selects on validation, evaluates test, and rebuilds this README, the report and notebook sources. It overwrites generated results. Keep the configuration fixed before final testing. CPU execution uses {config['threads']} PyTorch threads. Measured system: `{info['platform']}`; processor: `{info['processor']}`. Total logged fitting/validation time: {summary.train_seconds_mean.sum()+2*summary.loc[['GRU','GRU_calendar'],'train_seconds_mean'].sum():.1f} seconds. End-to-end runtime varies with network and hardware. Checkpoints and raw data remain local and are ignored by Git.

To run stages separately: `python scripts/get_data.py`, `python -m src.train`, `python -m src.evaluate`, and `python scripts/build_deliverables.py`. Notebook 01 performs EDA; notebook 02 can reproduce experiments or inspect saved results. Execute both with `python scripts/check_notebooks.py`. `results/metrics.csv` contains model, seed, horizon, metric and value.

For a historical viva demonstration: `python -m src.predict --origin 2016-07-01T12:00:00`. This prints saved-model forecasts and observed temperatures for the four horizons; it is not a live weather service.

The six-page report is [report/report.pdf](report/report.pdf). See [submission checklist](docs/SUBMISSION_CHECKLIST.md) and [viva guide](docs/VIVA_GUIDE.md). The ten-slide demo content is in `report/demo_content.json`; the editable PowerPoint is [report/demo.pptx](report/demo.pptx).

## AI-use statement

{ai}

## References

'''+"\n".join(f"- [{label}]({url})" for label,url in refs)+"\n- Supplied EC417 project brief (P22 and submission pages), dataset guide and updated allocation table.\n"
    (ROOT/"README.md").write_text(readme,encoding="utf-8")
    # Explicit page breaks make the report exactly six pages and keep charts readable.
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyProject",fontName="Helvetica",fontSize=10.5,leading=15,spaceAfter=10,textColor=colors.HexColor("#20334d")))
    styles["Title"].textColor=colors.HexColor("#12334b")
    styles["Heading1"].textColor=colors.HexColor("#087e8b")
    story=[]
    def p(text,style="BodyProject"):
        story.append(Paragraph(text,styles[style]))
    def heading(text): p(text,"Heading1")
    def chart(name,width=6.5):
        from PIL import Image as PILImage
        path=ROOT/f"results/figures/{name}.png"
        with PILImage.open(path) as im: ratio=im.height/im.width
        story.append(Image(str(path),width=width*inch,height=width*inch*ratio))
        story.append(Spacer(1,10))
    def page(): story.append(PageBreak())
    p("JenaClimateForecasting","Title")
    p("EC417 Introduction to Machine Intelligence - Term Project P22","Heading2")
    p("M Sanjhay Baargav - 623147<br/>Trinesh Kumar - 623138<br/>Department of Electronics and Communication Engineering<br/>National Institute of Technology Andhra Pradesh")
    heading("1. Problem and measured finding")
    p("We forecast air temperature 24 hours ahead from 120 hourly weather observations. The primary endpoint is test MAE in degrees Celsius. RMSE provides a secondary measure of large errors. A persistence baseline asks whether a learned model provides value beyond repeating the latest temperature.")
    p(headline+f" The lowest mean 24-hour MAE in this experiment belongs to {best}.")
    p("The experiment compares persistence, a regularized linear model on flattened windows, a GRU, and the same GRU with cyclical calendar inputs. Every method uses the same forecast origins and temperature targets. Seeds 17, 42 and 147 measure training variability for the neural models.")
    heading("Scope")
    p("All results come from the supplied Jena 2009-2016 benchmark. The main endpoint is 24 hours; additional fixed horizons of 1, 6 and 12 hours show how forecast difficulty changes. All learned models share these outputs, making calendar inputs the isolated proposed improvement.")
    p("The design and implementation follow the P22 task and the general submission requirements. No target accuracy is assumed, and a proposed improvement can perform worse.")
    page()
    heading("2. Data, EDA and leakage controls")
    p(f"The archive contains {audit['raw_rows']:,} records. The audit found {audit['duplicate_timestamps']} duplicate timestamps. Nonzero sentinel counts: "+", ".join(f"{k}: {v}" for k,v in audit['sentinel_counts'].items() if v)+".")
    p("Raw observations split into 2009-2014 training, 2015 validation and 2016 test. Hourly means use bins ending at the timestamp. Wind speed/direction become Cartesian components before averaging. Missing inputs are forward-filled within a period and then filled with training medians. Feature means and standard deviations use training data alone.")
    chart("temperature_split")
    p(f"Training has {audit['windows']['train']:,} windows at stride {config['train_stride']}. Validation has {audit['windows']['validation']:,} and test has {audit['windows']['test']:,} hourly windows. Each window contains 120 inputs. No window crosses a split boundary. Missing observed targets and missing persistence references are excluded.")
    p("The seasonal temperature cycle motivates cyclical features. The station download page states CC BY 4.0. The TensorFlow tutorial documents the exact historical mirror. Archive and CSV SHA-256 hashes, access timestamp and full source URLs are recorded in data/source_manifest.json.")
    page()
    heading("3. Models and controlled improvement")
    p(methods)
    for name in ["GRU","GRU_calendar"]:
        group=selection[selection.model==name]
        p(f"{name}: {int(group.parameters.iloc[0]):,} trainable parameters. Selected epochs by seed: "+", ".join(f"{int(r.seed)}: {int(r.best_epoch)}" for _,r in group.iterrows())+".")
    p("The calendar inputs are sin/cos of hour and sin/cos of within-year position, with leap years handled explicitly. They are bounded and deterministic. Main and improved models have equal hidden widths, output horizons, optimizer settings, data windows and seed lists. Their input parameter counts differ because four inputs are added.")
    chart("learning_curves")
    p("Learning curves show validation MAE at the primary 24-hour endpoint. Every run improves from its first epoch, and the selected checkpoints occur near the end of the fixed budget. This motivates testing longer training on validation data in future work, but does not prove it will help. Only validation results determine checkpoint choices.")
    page()
    heading("4. Final test evaluation")
    cells=[["Model","MAE (C)","RMSE (C)","Time (s)"]]
    for name,r in summary.iterrows():
        cells.append([name,f"{r.mae_mean:.3f} +/- {r.mae_std:.3f}",f"{r.rmse_mean:.3f} +/- {r.rmse_std:.3f}",f"{r.train_seconds_mean:.1f}"])
    t=Table(cells,colWidths=[123,126,126,70])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#12334b")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),("BOTTOMPADDING",(0,0),(-1,-1),10),("TOPPADDING",(0,0),(-1,-1),10),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#eef4f6"),colors.white])]))
    story.append(t); story.append(Spacer(1,12))
    p("Neural results show mean +/- sample standard deviation across three seeds. Deterministic baselines run once and have zero seed variability. This does not establish zero uncertainty about future weather. Times include fitting and validation selection; neural times are per-seed means.")
    chart("horizon_mae")
    p(headline+" Persistence peaks at 12 hours and improves at 24 hours, consistent with the daily temperature cycle. This is a descriptive comparison, not a significance claim. Overlapping forecast origins violate independent-sample assumptions, so naive per-window confidence intervals would be misleading.")
    page()
    heading("5. Forecast behaviour and error analysis")
    chart("two_test_weeks")
    p("The chart uses the first eligible two test weeks and the predeclared seed 17. It compares individual forecasts, not a seed ensemble, at identical target times.")
    p(error_text)
    chart("monthly_errors",width=5.8)
    p("The monthly plot averages absolute errors over seeds. The largest-error table contains individual seed-17 examples. Weather-front explanations remain hypotheses without additional atmospheric or spatial observations.")
    page()
    heading("6. Limitations, reproducibility and next steps")
    p(limitations)
    p("Next experiments should test variable-drop ablations and wider training budgets using validation data, then reserve a genuinely new final period. Rolling-origin evaluation would assess performance across multiple future periods. These are future work, not reported experiments.")
    p(f"Reproduction: Python 3.12; install requirements.txt, then run <b>python run_project.py</b>. Tests: <b>python -m unittest discover -s tests -v</b>. Measured platform: {html.escape(info['platform'])}. PyTorch uses {config['threads']} CPU threads. Full configuration, validation selections, metrics and audit records are saved. Runtime depends on hardware and download speed.")
    heading("AI-use statement")
    p(ai)
    heading("References")
    for label,url in refs:
        p(f'{html.escape(label)}. <link href="{url}" color="#087e8b">{url}</link>',"BodyText")
        story.append(Spacer(1,5))
    p("Course sources: EC417_TermProjects.pdf (P22 and submission pages), EC417_Dataset_Guide.md, and updated allocation table. Source pages accessed 29 October 2026.","BodyText")
    def footer(canvas,doc):
        canvas.setFont("Helvetica",8)
        canvas.setFillColor(colors.HexColor("#5a6a7c"))
        canvas.drawString(50,25,"EC417 P22  |  623147 + 623138  |  JenaClimateForecasting")
        canvas.drawRightString(545,25,str(doc.page))
    SimpleDocTemplate(str(ROOT/"report/report.pdf"),pagesize=(595.28,841.89),rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    build_notebooks()
    demo=[
        {"title":"JenaClimateForecasting","body":"P22 - EC417 Introduction to Machine Intelligence\nM Sanjhay Baargav (623147)\nTrinesh Kumar (623138)","notes":"Introduce the team and the forecasting question. 20 seconds."},
        {"title":"Forecasting task","body":"120 hourly observations predict temperature 24 hours ahead\nPrimary metric: MAE in degrees Celsius\nAuxiliary forecast horizons: 1, 6 and 12 hours","notes":"Explain forecast origin, history and target. 25 seconds."},
        {"title":"Chronological data split","body":f"{audit['raw_rows']:,} raw observations\nTrain: 2009-2014\nValidation: 2015\nFinal test: 2016","notes":"Source: https://www.bgc-jena.mpg.de/wetter/ and https://www.tensorflow.org/tutorials/structured_data/time_series. 30 seconds."},
        {"title":"Preprocessing without future information","body":"Hourly bins end at their timestamp\nWind components replace direction and speed\nFill gaps within each period\nFit normalization using training years\nCreate windows after splitting","notes":"Explain -9999 sentinels, missing-target exclusion, and split-boundary tests. 30 seconds."},
        {"title":"Baselines and GRU","body":f"Persistence repeats the latest temperature\nRidge uses the flattened 120-hour window\nGRU uses {config['hidden_size']} hidden units and four outputs\nValidation chooses penalties and checkpoints","notes":"Reference: https://arxiv.org/abs/1406.1078. Explain gates, not just the name. 35 seconds."},
        {"title":"Calendar feature experiment","body":"Add four inputs to the GRU\nSine and cosine of hour\nSine and cosine of position within the year\nKeep data, seeds and training rules fixed","notes":"The input parameter count increases slightly. Both GRUs already share the auxiliary horizon outputs. 30 seconds."},
        {"title":"Measured 24-hour test results","body":headline,"kind":"results","notes":"Read MAE means and standard deviations. Baselines are deterministic. No significance claim. 40 seconds."},
        {"title":"Error versus forecast horizon","body":"Longer horizons test a different prediction difficulty","kind":"horizon","notes":"Use the measured horizon chart. Error bars in the report show seed variability. 30 seconds."},
        {"title":"Forecast errors and limits","body":f"Largest calendar-GRU monthly MAE: {cold.mae:.3f} C in month {int(cold.month)}\nOne station and one test year\nOverlapping windows share weather conditions\nWeather causes of extreme errors remain hypotheses","notes":error_text+" 35 seconds."},
        {"title":"Reproduction and next steps","body":"python run_project.py\nReview the code and explain the results\nNext: variable ablation and new test years\nAI assistance: OpenAI Codex, disclosed in the report","notes":"Show the README, tests and source modules. Explain team verification and commit-history requirements. 25 seconds."}
    ]
    (ROOT/"report/demo_content.json").write_text(json.dumps(demo,indent=2),encoding="utf-8")
    print("Created README, report, notebooks and ten-slide demo content.")

def build_notebooks():
    setup="from pathlib import Path\nimport os, sys\nROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\nos.chdir(ROOT)\nsys.path.insert(0, str(ROOT))"
    eda=nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# P22 JenaClimateForecasting: EDA\nM Sanjhay Baargav (623147), Trinesh Kumar (623138).\nExplore training data before model selection. See data/README.md for provenance."),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_code_cell("from scripts.get_data import main\nmain()\nfrom src.data import load_hourly, TARGET\nparts, observed, mean, std, audit = load_hourly()\naudit"),
        nbf.v4.new_code_cell("parts['train'].describe().T"),
        nbf.v4.new_code_cell("import matplotlib.pyplot as plt\nparts['train'][TARGET].resample('7D').mean().plot(figsize=(12,4),title='Training temperature: weekly mean')\nplt.ylabel('Temperature (degrees C)')\nplt.show()"),
        nbf.v4.new_code_cell("parts['train'][TARGET].hist(bins=50,figsize=(8,4))\nplt.xlabel('Temperature (degrees C)')\nplt.ylabel('Hourly observations')\nplt.show()"),
        nbf.v4.new_code_cell("corr=parts['train'].corr()\nfig,ax=plt.subplots(figsize=(10,8))\nim=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')\nax.set_xticks(range(len(corr)),corr.columns,rotation=90)\nax.set_yticks(range(len(corr)),corr.columns)\nfig.colorbar(im,ax=ax)\nplt.tight_layout()\nplt.show()"),
        nbf.v4.new_markdown_cell("## Interpretation\nInspect annual variation, extreme temperatures, correlated thermodynamic variables and invalid wind counts. Fit all statistics on training years only. Test-period plots are for final reporting after selection, not feature tuning.")])
    exp=nbf.v4.new_notebook(cells=[
        nbf.v4.new_markdown_cell("# P22 experiments and results\nThis notebook runs top to bottom. It uses saved results when available. Set RETRAIN=True to regenerate every experiment; if outputs are absent, it runs the full pipeline automatically."),
        nbf.v4.new_code_cell(setup),
        nbf.v4.new_code_cell("import subprocess\nRETRAIN = False\nif RETRAIN or not (ROOT/'results/summary.csv').exists():\n    subprocess.run([sys.executable,'run_project.py'],check=True)"),
        nbf.v4.new_code_cell("import pandas as pd\npd.read_csv('results/summary.csv')"),
        nbf.v4.new_code_cell("from IPython.display import display, Image\nfor name in ['horizon_mae','two_test_weeks','learning_curves','training_curves','monthly_errors']:\n    display(Image(filename=f'results/figures/{name}.png'))"),
        nbf.v4.new_code_cell("pd.read_csv('results/largest_errors.csv')"),
        nbf.v4.new_code_cell("subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],check=True)"),
        nbf.v4.new_markdown_cell("## Discussion\nUse the measured MAE change to state whether calendar features helped. Seed spread measures optimization variability only. Review the largest errors and limitations before the viva. AI assistance is disclosed in README.md.")])
    for name,nb in [("01_eda",eda),("02_experiments",exp)]:
        nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
        nbf.write(nb,ROOT/f"notebooks/{name}.ipynb")

if __name__=="__main__":
    main()
