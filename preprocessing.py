import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NHSDataPreprocessor:
    def __init__(self, input_path='data/raw/sample_data.csv', output_path='data/processed'):
        self.input_path = input_path
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.df = None
        
    def load_data(self):
        """Load and perform initial data checks"""
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        logger.info(f"Loaded {len(self.df)} records with {len(self.df.columns)} features")
        
        # Initial data quality checks
        missing_values = self.df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
            
        return self
    
    def process_dates(self):
        """Process date columns and create temporal features"""
        logger.info("Processing date features")
        
        # Convert to datetime
        date_columns = ['admission_date', 'discharge_date']
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col])
        
        # Create temporal features
        self.df['admission_month'] = self.df['admission_date'].dt.month
        self.df['admission_dayofweek'] = self.df['admission_date'].dt.dayofweek
        self.df['is_weekend'] = self.df['admission_dayofweek'].isin([5, 6]).astype(int)
        
        # Verify length of stay
        calculated_los = (self.df['discharge_date'] - self.df['admission_date']).dt.days
        if not (calculated_los == self.df['length_of_stay']).all():
            logger.warning("Discrepancy found in length of stay calculations")
            
        return self
    
    def create_patient_history(self):
        """Generate patient history features"""
        logger.info("Generating patient history features")
        
        # Sort by patient and date
        self.df = self.df.sort_values(['patient_id', 'admission_date'])
        
        # Calculate previous admissions
        self.df['previous_admissions'] = self.df.groupby('patient_id').cumcount()
        
        # Time since last admission
        self.df['days_since_last_admission'] = self.df.groupby('patient_id')['admission_date'].diff().dt.days
        
        # Total admissions per patient
        admission_counts = self.df.groupby('patient_id').size()
        self.df['total_admissions'] = self.df['patient_id'].map(admission_counts)
        
        return self
    
    def encode_categorical(self):
        """Encode categorical variables"""
        logger.info("Encoding categorical features")
        
        # Label encode binary/categorical features
        le = LabelEncoder()
        self.df['gender_encoded'] = le.fit_transform(self.df['gender'])
        self.df['specialty_encoded'] = le.fit_transform(self.df['specialty'])
        
        # Create dummy variables for discharge destination
        self.df = pd.get_dummies(
            self.df, 
            columns=['discharge_destination'], 
            prefix='discharge_to'
        )
        
        # Create condition flags
        conditions = {
            'I21': 'acute_mi', 'I50': 'heart_failure', 'J44': 'copd',
            'E11': 'diabetes', 'N18': 'kidney_disease', 'I10': 'hypertension',
            'F32': 'depression', 'K92': 'gi_bleeding', 'J18': 'pneumonia',
            'I63': 'stroke'
        }
        
        for code, name in conditions.items():
            self.df[f'has_{name}'] = (
                (self.df['primary_diagnosis_icd10'] == code) | 
                (self.df['comorbidities_icd10'].str.contains(code, na=False))
            ).astype(int)
            
        return self
    
    def create_visualizations(self):
        """Generate EDA visualizations"""
        logger.info("Creating EDA visualizations")
        
        # Set style
        plt.style.use('default')
        
        # Create visualization directory
        viz_path = self.output_path / 'visualizations'
        viz_path.mkdir(exist_ok=True)
        
        # 1. Readmission rates by feature
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Readmission Analysis', fontsize=16)
        
        # Age groups
        age_bins = [0, 30, 50, 70, 100]
        age_labels = ['0-30', '31-50', '51-70', '70+']
        self.df['age_group'] = pd.cut(self.df['age'], bins=age_bins, labels=age_labels)
        sns.barplot(data=self.df, x='age_group', y='readmitted_30_days', ax=axes[0,0])
        axes[0,0].set_title('Readmission Rate by Age Group')
        
        # Length of stay
        sns.boxplot(data=self.df, x='readmitted_30_days', y='length_of_stay', ax=axes[0,1])
        axes[0,1].set_title('Length of Stay by Readmission Status')
        
        # Emergency admission
        sns.barplot(data=self.df, x='emergency_admission', y='readmitted_30_days', ax=axes[1,0])
        axes[1,0].set_title('Readmission Rate by Admission Type')
        
        # Deprivation quintile
        sns.barplot(data=self.df, x='deprivation_quintile', y='readmitted_30_days', ax=axes[1,1])
        axes[1,1].set_title('Readmission Rate by Deprivation Quintile')
        
        plt.tight_layout()
        plt.savefig(viz_path / 'readmission_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap of numerical features
        numerical_cols = ['age', 'length_of_stay', 'previous_admissions', 
                         'total_admissions', 'days_since_last_admission']
        corr_matrix = self.df[numerical_cols + ['readmitted_30_days']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.savefig(viz_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self
    
    def prepare_final_features(self):
        """Prepare final feature set for modeling"""
        logger.info("Preparing final feature set")
        
        feature_cols = [
            # Patient demographics
            'age', 'gender_encoded', 'deprivation_quintile',
            
            # Admission details
            'length_of_stay', 'emergency_admission', 'specialty_encoded',
            'admission_month', 'admission_dayofweek', 'is_weekend',
            
            # Patient history
            'previous_admissions', 'total_admissions', 'days_since_last_admission',
            
            # Condition flags
            'has_acute_mi', 'has_heart_failure', 'has_copd', 'has_diabetes',
            'has_kidney_disease', 'has_hypertension', 'has_depression',
            'has_gi_bleeding', 'has_pneumonia', 'has_stroke',
            
            # Discharge destination
            'discharge_to_Home', 'discharge_to_Care Home',
            'discharge_to_Another Hospital', 'discharge_to_Rehabilitation Unit'
        ]
        
        # Create final features and target
        X = self.df[feature_cols]
        y = self.df['readmitted_30_days']
        
        # Save processed data
        X.to_csv(self.output_path / 'X_preprocessed.csv', index=False)
        y.to_csv(self.output_path / 'y_preprocessed.csv', index=False)
        
        logger.info(f"Saved preprocessed data with {len(X.columns)} features")
        return X, y
    
    def run_pipeline(self):
        """Run the complete preprocessing pipeline"""
        try:
            (self.load_data()
                 .process_dates()
                 .create_patient_history()
                 .encode_categorical()
                 .create_visualizations())
            
            X, y = self.prepare_final_features()
            logger.info("Preprocessing pipeline completed successfully!")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    preprocessor = NHSDataPreprocessor()
    X, y = preprocessor.run_pipeline()
    
    print("\nPreprocessing Summary:")
    print(f"Total samples: {len(X)}")
    print(f"Total features: {len(X.columns)}")
    print(f"Readmission rate: {y.mean():.2%}")
    print("\nFeature list:")
    print("\n".join(f"- {col}" for col in X.columns))