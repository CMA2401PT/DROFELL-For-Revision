# Training Records and Plotting

Parent training entry points create subdirectories such as `common/`, `err-eps/`, and `bfloat16/`, containing `record.json` and checkpoints.

- `plot.py` reads only `err-eps/record.json` by default and plots loss, speed, and GPU memory on two axes.
- `plot_multi.py` reads fixed `err-eps`, `bfloat16_start`, and `bfloat16` outputs for curve comparison.

Both depend on the current directory and use notebook-style code:

```bash
cd scripts/qllm/train/records
PYTHONPATH=../../../.. MPLBACKEND=Agg "${DROFELL_PY:-python}" plot.py
```

`plot.py` calls `plt.show()`; `plot_multi.py` constructs figures without one save path. The scripts also fix step buckets, axis ranges, and a four-step speed window. Before new use, verify `record.json` schema, sample interval, and batch size and add the desired `savefig` target.
