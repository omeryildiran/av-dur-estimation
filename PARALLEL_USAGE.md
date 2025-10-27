# Running runFitting.py in Parallel

The script has been updated to process multiple data files in parallel across multiple CPU cores for faster execution.

## Command Line Arguments

```bash
python runFitting.py <data_files> <model> <simulations> <optimizer> <starts> <integration> <cores>
```

### Arguments:
1. **data_files** (required): Comma-separated list of CSV files
2. **model** (optional, default: "lognorm"): Model type ("lognorm" or "gaussian")
3. **simulations** (optional, default: 500): Number of Monte Carlo simulations
4. **optimizer** (optional, default: "bads"): Optimization method
5. **starts** (optional, default: 1): Number of random starts
6. **integration** (optional, default: "analytical"): Integration method ("analytical" or "numerical")
7. **cores** (optional, default: auto): Number of CPU cores to use

## Usage Examples

### Example 1: Use all available cores (automatic)
```bash
python runFitting.py "as_all.csv,oy_all.csv,dt_all.csv,HH_all.csv,ip_all.csv,ln_all.csv" "lognorm" 500 "bads" 5
```

### Example 2: Specify number of cores (e.g., 4 cores)
```bash
python runFitting.py "as_all.csv,oy_all.csv,dt_all.csv,HH_all.csv" "lognorm" 500 "bads" 5 "analytical" 4
```

### Example 3: Use 8 cores for large batch
```bash
python runFitting.py "as_all.csv,oy_all.csv,dt_all.csv,HH_all.csv,ip_all.csv,ln_all.csv,LN01_all.csv,mh_all.csv,ml_all.csv,mt_all.csv,qs_all.csv,sx_all.csv" "lognorm" 500 "bads" 5 "analytical" 8
```

### Example 4: Single core (sequential processing)
```bash
python runFitting.py "mt_all.csv,as_all.csv" "lognorm" 500 "bads" 5 "analytical" 1
```

## How It Works

1. **Parallel Processing**: The script uses Python's `multiprocessing.Pool` to process multiple data files simultaneously
2. **Automatic Core Detection**: By default, it uses all available CPU cores (but not more than the number of files)
3. **Error Handling**: Each file is processed independently with error catching, so one failed file won't stop the others
4. **Summary Report**: After completion, you get a summary showing:
   - Total processing time
   - Average time per file
   - List of successful files (✅)
   - List of failed files (❌) with error messages

## Performance Tips

- **Optimal cores**: Use as many cores as you have files (or as many as your CPU has)
- **Memory consideration**: Each core will load data independently, so monitor RAM usage
- **I/O bottleneck**: If you have an SSD, parallel processing will be much more effective
- **CPU-bound tasks**: Since model fitting is CPU-intensive, you'll see significant speedup with more cores

## Expected Speedup

With **N cores** and **N files**, you can expect close to **N times faster** execution compared to sequential processing (assuming sufficient CPU and memory resources).

For example:
- 12 files on 1 core: ~12 hours
- 12 files on 12 cores: ~1 hour (ideal case)
- 12 files on 6 cores: ~2 hours

## Check Your CPU Cores

To see how many cores you have:

```bash
# macOS/Linux
python -c "import multiprocessing; print(f'CPU cores: {multiprocessing.cpu_count()}')"

# Or check with system command
sysctl -n hw.ncpu  # macOS
nproc              # Linux
```
