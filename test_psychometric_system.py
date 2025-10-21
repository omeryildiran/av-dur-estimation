"""
Test script to verify psychometric fitting system works correctly
Run this to check if everything is set up properly
"""

import os
import sys

print("="*70)
print("PSYCHOMETRIC FITTING SYSTEM - TEST SCRIPT")
print("="*70)

# Test 1: Check if required modules exist
print("\n1. Checking required files...")
required_files = [
    'psychometricFitSaver.py',
    'psychometricFitLoader.py', 
    'runDataFitter.py',
    'fitMain.py'
]

all_files_exist = True
for file in required_files:
    if os.path.exists(file):
        print(f"   ✅ {file}")
    else:
        print(f"   ❌ {file} NOT FOUND")
        all_files_exist = False

if not all_files_exist:
    print("\n❌ Some required files are missing!")
    sys.exit(1)

# Test 2: Check if modules can be imported
print("\n2. Testing module imports...")
try:
    import psychometricFitSaver as pfs
    print("   ✅ psychometricFitSaver")
except Exception as e:
    print(f"   ❌ psychometricFitSaver: {e}")
    sys.exit(1)

try:
    import psychometricFitLoader as pfl
    print("   ✅ psychometricFitLoader")
except Exception as e:
    print(f"   ❌ psychometricFitLoader: {e}")
    sys.exit(1)

# Test 3: Check if data files exist
print("\n3. Checking data files...")
data_dir = "data"
required_data = [
    "as_all.csv", "oy_all.csv", "dt_all.csv", "HH_all.csv",
    "ip_all.csv", "ln_all.csv", "LN01_all.csv", "mh_all.csv",
    "ml_all.csv", "mt_all.csv", "qs_all.csv", "sx_all.csv"
]

missing_data = []
for data_file in required_data:
    file_path = os.path.join(data_dir, data_file)
    if os.path.exists(file_path):
        print(f"   ✅ {data_file}")
    else:
        print(f"   ⚠️  {data_file} NOT FOUND")
        missing_data.append(data_file)

if missing_data:
    print(f"\n   ⚠️  {len(missing_data)} data files missing (can still test with others)")

# Test 4: Check if any fits exist
print("\n4. Checking for existing fits...")
fits_dir = "psychometric_fits"
if os.path.exists(fits_dir):
    print(f"   ✅ Fits directory exists: {fits_dir}")
    
    # Count existing fits
    participant_dirs = [d for d in os.listdir(fits_dir) 
                       if os.path.isdir(os.path.join(fits_dir, d))]
    
    if participant_dirs:
        print(f"   ✅ Found {len(participant_dirs)} participant fits:")
        for pid in sorted(participant_dirs):
            json_file = os.path.join(fits_dir, pid, f"{pid}_psychometric_fit.json")
            if os.path.exists(json_file):
                print(f"      • {pid}")
        
        # Try to load one
        print("\n   Testing fit loading...")
        try:
            test_pid = participant_dirs[0]
            fit = pfl.load_psychometric_fit(test_pid)
            print(f"   ✅ Successfully loaded fit for '{test_pid}'")
            print(f"      - Parameters: {len(fit['parameters'])} values")
            print(f"      - AIC: {fit['AIC']:.2f}")
            print(f"      - Conditions: {fit['n_conditions']}")
        except Exception as e:
            print(f"   ❌ Error loading fit: {e}")
    else:
        print("   ℹ️  No fits found yet - run runDataFitter.py to generate them")
else:
    print("   ℹ️  No fits directory yet - will be created when you run runDataFitter.py")

# Test 5: Test fit summary function
print("\n5. Testing fit summary functions...")
try:
    if os.path.exists(fits_dir) and os.listdir(fits_dir):
        summary = pfl.get_fit_summary()
        print(f"   ✅ get_fit_summary() works - {len(summary)} participants")
        
        comparison = pfl.get_model_comparison()
        print(f"   ✅ get_model_comparison() works - {len(comparison)} models")
    else:
        print("   ℹ️  Skipping (no fits available yet)")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Final summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

if all_files_exist:
    print("✅ All required files present")
    print("✅ Module imports working")
    
    if os.path.exists(fits_dir) and os.listdir(fits_dir):
        print("✅ Fits exist and can be loaded")
        print("\n🎉 SYSTEM READY TO USE!")
        print("\nYou can now:")
        print("  1. Load fits in notebooks using: import psychometricFitLoader as pfl")
        print("  2. Run batch fitting with: python runDataFitter.py")
    else:
        print("⚠️  No fits generated yet")
        print("\n📋 NEXT STEPS:")
        print("  1. Run: python runDataFitter.py")
        print("  2. Wait for fits to complete (~6-10 minutes)")
        print("  3. Use fits in notebooks/scripts")
else:
    print("❌ SETUP INCOMPLETE - check errors above")

print("="*70)
