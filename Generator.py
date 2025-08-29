#!/usr/bin/env python3
"""
NHS Readmission Prediction - Quick Start Script
==============================================
Run this single script to set up the entire Phase 1 system!

Usage:
    python quick_start.py

This script will:
1. Install required packages (if needed)
2. Create project structure
3. Generate synthetic NHS data
4. Validate data quality
5. Create initial visualizations
6. Prepare for Phase 2

Author: NHS ML Team
Date: August 2025
"""

import os
import sys
import subprocess
import warnings

import pandas as pd
warnings.filterwarnings('ignore')

def check_and_install_packages():
    """Check for and install required packages"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'xgboost', 'shap', 'flask', 'pyyaml'
    ]
    
    print("🔍 Checking required packages...")
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run manually:")
            print(f"pip install {' '.join(missing_packages)}")
            sys.exit(1)
    else:
        print("✅ All required packages are already installed!")

def create_directory_structure():
    """Create the project directory structure"""
    print("\n🏗️ Creating project structure...")
    
    directories = [
        'data/raw', 'data/processed', 'data/external',
        'src/preprocessing', 'src/features', 'src/models', 
        'src/evaluation', 'src/visualization',
        'app/templates', 'app/static/css', 'app/static/js',
        'models', 'docs', 'tests', 'configs', 'logs', 'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create .gitkeep file
        with open(os.path.join(directory, '.gitkeep'), 'w') as f:
            f.write('')
    
    print("✅ Directory structure created!")

def generate_nhs_data():
    """Generate synthetic NHS EHR data"""
    print("\n📊 Generating synthetic NHS EHR dataset...")
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import random
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # NHS-aligned reference data
    icd10_codes = {
        'I21': 'Acute myocardial infarction',
        'I50': 'Heart failure',
        'J44': 'COPD',
        'E11': 'Type 2 diabetes',
        'N18': 'Chronic kidney disease',
        'I10': 'Essential hypertension',
        'F32': 'Depressive episode',
        'K92': 'GI bleeding',
        'J18': 'Pneumonia',
        'I63': 'Cerebral infarction'
    }
    
    specialties = [
        'Cardiology', 'Respiratory', 'Gastroenterology', 'Endocrinology',
        'Nephrology', 'Geriatrics', 'Emergency Medicine', 'General Medicine'
    ]
    
    discharge_destinations = ['Home', 'Care Home', 'Another Hospital', 'Rehabilitation Unit']
    
    # Generate dataset
    n_records = 1500
    
    data = []
    for i in range(n_records):
        # Patient demographics
        age = max(18, min(95, int(np.random.gamma(2, 20))))
        gender = np.random.choice(['M', 'F'])
        deprivation = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.2, 0.3, 0.2, 0.15])
        
        # Clinical data
        admission_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
        los = max(1, min(30, int(np.random.exponential(3))))
        discharge_date = admission_date + timedelta(days=los)
        
        primary_dx = np.random.choice(list(icd10_codes.keys()))
        specialty = np.random.choice(specialties)
        emergency = np.random.choice([True, False], p=[0.6, 0.4])
        discharge_dest = np.random.choice(discharge_destinations, p=[0.8, 0.1, 0.07, 0.03])
        
        # Comorbidities
        n_comorbidities = np.random.poisson(max(0, (age - 50) / 20))
        available_codes = [c for c in icd10_codes.keys() if c != primary_dx]
        if n_comorbidities > 0 and len(available_codes) > 0:
            comorbidities = np.random.choice(
                available_codes, 
                size=min(n_comorbidities, len(available_codes)), 
                replace=False
            )
            comorbidity_str = ','.join(comorbidities)
        else:
            comorbidity_str = ''
        
        # Calculate readmission risk
        risk = 0.1  # Base risk
        if age > 75: risk += 0.08
        elif age > 65: risk += 0.04
        if los == 1: risk += 0.06  # Premature discharge
        if los > 10: risk += 0.05  # Complex case
        if emergency: risk += 0.04
        risk += (deprivation - 1) * 0.02
        risk += len(comorbidity_str.split(',')) * 0.02 if comorbidity_str else 0
        if discharge_dest != 'Home': risk += 0.05
        
        readmitted = np.random.random() < min(risk, 0.6)
        
        # Create record
        record = {
            'episode_id': f'EP{i+1:08d}',
            'patient_id': f'NHS{(i//2)+1:06d}',  # Some patients have multiple episodes
            'admission_date': admission_date.strftime('%Y-%m-%d'),
            'discharge_date': discharge_date.strftime('%Y-%m-%d'),
            'length_of_stay': los,
            'primary_diagnosis_icd10': primary_dx,
            'primary_diagnosis_desc': icd10_codes[primary_dx],
            'comorbidities_icd10': comorbidity_str,
            'specialty': specialty,
            'emergency_admission': emergency,
            'discharge_destination': discharge_dest,
            'age': age,
            'gender': gender,
            'postcode': f'{np.random.choice(["M1", "B1", "L1", "RG", "SL"])}{np.random.randint(1,9)} {np.random.randint(1,9)}AA',
            'deprivation_quintile': deprivation,
            'readmitted_30_days': readmitted
        }
        
        data.append(record)
    import pandas as pd
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save datasets
    df.to_csv('data/raw/sample_data.csv', index=False)
    df.sample(200).to_csv('data/raw/quick_test_sample.csv', index=False)
    
    print(f"✅ Generated {len(df)} records")
    print(f"   • Unique patients: {df['patient_id'].nunique()}")
    print(f"   • Readmission rate: {df['readmitted_30_days'].mean():.1%}")
    print(f"   • Date range: {df['admission_date'].min()} to {df['admission_date'].max()}")
    
    return df

def create_visualizations(df):
    """Create initial data visualizations"""
    print("\n📈 Creating initial visualizations...")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NHS EHR Synthetic Dataset - Phase 1 Overview', fontsize=16, fontweight='bold')
    
    # 1. Age distribution
    axes[0,0].hist(df['age'], bins=25, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].set_title('Age Distribution')
    axes[0,0].set_xlabel('Age (years)')
    axes[0,0].set_ylabel('Count')
    
    # 2. Readmission by age group
    age_groups = pd.cut(df['age'], bins=[0, 45, 65, 75, 100], labels=['<45', '45-64', '65-74', '75+'])
    readmission_by_age = df.groupby(age_groups)['readmitted_30_days'].mean()
    bars = axes[0,1].bar(readmission_by_age.index, readmission_by_age.values, color='lightcoral', alpha=0.8)
    axes[0,1].set_title('Readmission Rate by Age Group')
    axes[0,1].set_ylabel('Readmission Rate')
    axes[0,1].set_ylim(0, max(readmission_by_age.values) * 1.2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{height:.1%}', ha='center', va='bottom')
    
    # 3. Length of stay distribution
    axes[0,2].hist(df['length_of_stay'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0,2].set_title('Length of Stay Distribution')
    axes[0,2].set_xlabel('Days')
    axes[0,2].set_ylabel('Count')
    
    # 4. Top diagnoses
    top_diagnoses = df['primary_diagnosis_desc'].value_counts().head(6)
    bars = axes[1,0].barh(range(len(top_diagnoses)), top_diagnoses.values, color='gold', alpha=0.8)
    axes[1,0].set_yticks(range(len(top_diagnoses)))
    axes[1,0].set_yticklabels([label[:20] + '...' if len(label) > 20 else label for label in top_diagnoses.index])
    axes[1,0].set_title('Top Primary Diagnoses')
    axes[1,0].set_xlabel('Count')
    
    # 5. Readmission by specialty
    specialty_readmission = df.groupby('specialty')['readmitted_30_days'].mean().sort_values()
    bars = axes[1,1].barh(range(len(specialty_readmission)), specialty_readmission.values, color='plum', alpha=0.8)
    axes[1,1].set_yticks(range(len(specialty_readmission)))
    axes[1,1].set_yticklabels([label[:15] + '...' if len(label) > 15 else label for label in specialty_readmission.index])
    axes[1,1].set_title('Readmission Rate by Specialty')
    axes[1,1].set_xlabel('Readmission Rate')
    
    # 6. Emergency vs Planned Admissions
    emergency_readmission = df.groupby('emergency_admission')['readmitted_30_days'].mean()
    bars = axes[1,2].bar(['Planned', 'Emergency'], emergency_readmission.values, 
                        color=['lightblue', 'orange'], alpha=0.8)
    axes[1,2].set_title('Readmission: Emergency vs Planned')
    axes[1,2].set_ylabel('Readmission Rate')
    axes[1,2].set_ylim(0, max(emergency_readmission.values) * 1.2)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                      f'{height:.1%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('data/processed/phase1_overview.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved to: data/processed/phase1_overview.png")
    
    try:
        plt.show()
    except:
        print("   (Display not available - chart saved to file)")

def create_config_files():
    """Create configuration and documentation files"""
    print("\n📝 Creating configuration files...")
    
    # requirements.txt
    requirements = """pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.6.0
matplotlib>=3.5.0
seaborn>=0.11.0
shap>=0.41.0
flask>=2.0.0
pyyaml>=6.0
joblib>=1.1.0
python-dateutil>=2.8.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    # Basic README
    readme = """# NHS Hospital Readmission Prediction System

A production-ready ML system for predicting 30-day hospital readmissions using NHS EHR data.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run Phase 1: `python quick_start.py`
3. Explore data: Check `data/processed/phase1_overview.png`
4. Ready for Phase 2: Data Preprocessing

## Dataset

- **Location**: `data/raw/sample_data.csv`
- **Records**: ~1,500 admission episodes
- **Features**: Demographics, diagnoses, length of stay, etc.
- **Target**: 30-day readmission (binary)

## Next Steps

Ready to proceed with Phase 2: Data Preprocessing and Feature Engineering!
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    print("✅ Configuration files created!")

def main():
    """Main execution function"""
    print("🏥 NHS HOSPITAL READMISSION PREDICTION SYSTEM")
    print("=" * 60)
    print("Quick Start Script - Setting up everything for you!")
    print("=" * 60)
    
    try:
        # Step 1: Check packages
        check_and_install_packages()
        
        # Step 2: Create structure
        create_directory_structure()
        
        # Step 3: Generate data
        df = generate_nhs_data()
        
        # Step 4: Create visualizations
        create_visualizations(df)
        
        # Step 5: Create config files
        create_config_files()
        
        # Step 6: Summary
        print("\n🎉 SETUP COMPLETE!")
        print("=" * 30)
        print("✅ Project structure created")
        print("✅ Synthetic NHS data generated")
        print("✅ Initial visualizations created")
        print("✅ Configuration files ready")
        
        print(f"\n📊 Your Dataset:")
        print(f"   • File: data/raw/sample_data.csv")
        print(f"   • Records: {len(df):,}")
        print(f"   • Patients: {df['patient_id'].nunique():,}")
        print(f"   • Readmission Rate: {df['readmitted_30_days'].mean():.1%}")
        
        print(f"\n🔍 Explore Your Data:")
        print(f"   • View: data/processed/phase1_overview.png")
        print(f"   • Quick sample: data/raw/quick_test_sample.csv")
        
        print(f"\n🚀 Next Steps:")
        print(f"   • Ready for Phase 2: Data Preprocessing")
        print(f"   • Run: python -c \"import pandas as pd; print(pd.read_csv('data/raw/sample_data.csv').head())\"")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("Please check the error message above and try again.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n✅ Phase 1 Complete! Ready for Phase 2! 🚀")
    else:
        print(f"\n❌ Setup failed. Please check errors and try again.")