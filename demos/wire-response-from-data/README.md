# Wire response from data (demo)

Closed-loop test of learning a field response from laser data on ProtoDUNE-HD's
"bad APA" (WCT anode 0, floating W plane): simulate laser ADC data with the
Garfield best model of that APA, then recover it starting from a deliberately
perturbed model.  See `design.md` for the plan and status.

    source env.sh && bash scripts/check-env.sh   # environment report
    snakemake -c4 -n                             # what would run
    snakemake -c1 note                           # build note/wrfd.pdf
