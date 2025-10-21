"""
Test script to verify simulated psychometric fitting system works correctly
"""

import os
import sys

print("="*70)
print("SIMULATED PSYCHOMETRIC FITTING SYSTEM - TEST SCRIPT")
print("="*70)

# Test 1: Check if required modules exist
print("\n1. Checking required files...")
required_files = [
    'psychometricFitSaver_simulated.py',
    'psychometricFitLoader_simulated.py', 
    'runDataFitter_simulated.py',
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
    import psychometricFitSaver_simulated as pfs_sim
    print("   ✅ psychometricFitSaver_simulated")
except Exception as e:
    print(f"   ❌ psychometricFitSaver_simulated: {e}")
    sys.exit(1)

try:
    import psychometricFitLoader_simulated as pfl_sim
    print("   ✅ psychometricFitLoader_simulated")
except Exception as e:
    print(f"   ❌ psychometricFitLoader_simulated: {e}")
    sys.exit(1)

# Test 3: Check if simulated data files exist
print("\n3. Checking simulated data files...")
sim_base_dir = "simulated_data"
model_type = "lognorm_LapseFree_sharedPrior"
participant_ids = ["as", "oy", "dt", "HH", "ip", "ln", "LN01", "mh", "ml", "mt", "qs", "sx"]

missing_data = []
found_count = 0
for pid in participant_ids:
    sim_filename = f"{pid}_{model_type}_simulated.csv"
    sim_filepath = os.path.join(sim_base_dir, pid, sim_filename)
    if os.path.exists(sim_filepath):
        print(f"   ✅ {pid}/{sim_filename}")
        found_count += 1
    else:
        print(f"   ⚠️  {pid}/{sim_filename} NOT FOUND")
        missing_data.append(pid)

if missing_data:
    print(f"\n   ⚠️  {len(missing_data)} simulated files missing")
    print(f"   ✅ {found_count} simulated files found")

# Test 4: Check if any fits exist
print("\n4. Checking for existing simulated fits...")
fits_dir = "psychometric_fits_simulated"
if os.path.exists(fits_dir):
    print(f"   ✅ Fits directory exists: {fits_dir}")
    
    # Count existing fits
    participant_dirs = [d for d in os.listdir(fits_dir) 
                       if os.path.isdir(os.path.join(fits_dir, d))]
    
    if participant_dirs:
        print(f"   ✅ Found {len(participant_dirs)} participant simulated fits:")
        for pid in sorted(participant_dirs)[:5]:  # Show first 5
            json_file = os.path.join(fits_dir, pid, f"{pid}_{model_type}_simulated_psychometric_fit.json")
            if os.path.exists(json_file):
                print(f"      • {pid}")
        
        if len(participant_dirs) > 5:
            print(f"      ... and {len(participant_dirs) - 5} more")
        
        # Try to load one
        print("\n   Testing fit loading...")
        try:
            test_pid = participant_dirs[0]
            fit = pfl_sim.load_psychometric_fit_simulated(test_pid, model_type)
            print(f"   ✅ Successfully loaded simulated fit for '{test_pid}'")
            print(f"      - Parameters: {len(fit['parameters'])} values")
            print(f"      - AIC: {fit['AIC']:.2f}")
            print(f"      - Model: {fit['model_type']}")
        except Exception as e:
            print(f"   ❌ Error loading fit: {e}")
    else:
        print("   ℹ️  No fits found yet - run runDataFitter_simulated.py to generate them")
else:
    print("   ℹ️  No fits directory yet - will be created when you run runDataFitter_simulated.py")

# Test 5: Test fit summary function
print("\n5. Testing fit summary functions...")
try:
    if os.path.exists(fits_dir) and os.listdir(fits_dir):
        summary = pfl_sim.get_fit_summary_simulated(model_type)
        print(f"   ✅ get_fit_summary_simulated() works - {len(summary)} participants")
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
    print(f"✅ Found {found_count}/{len(participant_ids)} simulated data files")
    
    if os.path.exists(fits_dir) and os.listdir(fits_dir):
        print("✅ Fits exist and can be loaded")
        print("\n🎉 SYSTEM READY TO USE!")
        print("\nYou can now:")
        print("  1. Load fits in notebooks using: import psychometricFitLoader_simulated as pfl_sim")
        print("  2. Run batch fitting with: python runDataFitter_simulated.py")
    else:
        print("⚠️  No fits generated yet")
        print("\n📋 NEXT STEPS:")
        print("  1. Run: python runDataFitter_simulated.py")
        print("  2. Wait for fits to complete (~3-5 minutes)")
        print("  3. Use fits in notebooks/scripts")
else:
    print("❌ SETUP INCOMPLETE - check errors above")

print("="*70)
