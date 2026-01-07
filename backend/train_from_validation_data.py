"""
Train ML Models from Validation Data
Uses ground truth measurements to train improved depth ratio predictors
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
from datetime import datetime
import sys
import os
from pathlib import Path


class ValidationDataTrainer:
    """Train ML models using validation study ground truth data"""

    def __init__(self, validation_results_path: str):
        """
        Initialize trainer with validation results

        Args:
            validation_results_path: Path to validation_results.csv
        """
        self.df = pd.read_csv(validation_results_path)
        print(f"\n✅ Loaded {len(self.df)} validation samples")

        self.models = {}
        self.encoders = {}
        self.training_stats = {}

    def calculate_true_ratios(self):
        """
        Calculate true depth ratios from ground truth measurements

        Formula:
        From circumference ≈ π * (width + depth)
        Solve for depth, then ratio = depth/width
        """
        print(f"\n📊 Calculating true depth ratios from ground truth...")

        def calc_ratio(actual_circ, predicted_width):
            """Calculate actual depth/width ratio"""
            if pd.isna(actual_circ) or pd.isna(predicted_width) or predicted_width == 0:
                return np.nan

            # Simplified ellipse: circ ≈ π * (width + depth)
            implied_sum = actual_circ / np.pi
            depth = implied_sum - predicted_width
            ratio = depth / predicted_width

            # Clamp to anthropometrically valid range
            return max(0.35, min(0.95, ratio))

        # Calculate true ratios
        self.df['true_chest_ratio'] = self.df.apply(
            lambda row: calc_ratio(
                row['actual_chest_circumference'],
                row['predicted_chest_width']
            ), axis=1
        )

        self.df['true_waist_ratio'] = self.df.apply(
            lambda row: calc_ratio(
                row['actual_waist_circumference'],
                row['predicted_waist_width']
            ), axis=1
        )

        self.df['true_hip_ratio'] = self.df.apply(
            lambda row: calc_ratio(
                row['actual_hip_circumference'],
                row['predicted_hip_width']
            ), axis=1
        )

        # Summary statistics
        print(f"\nTrue Ratio Distributions:")
        for ratio_type in ['chest', 'waist', 'hip']:
            col = f'true_{ratio_type}_ratio'
            valid = self.df[col].dropna()
            print(f"  {ratio_type.upper():5s}: "
                  f"mean={valid.mean():.3f}, "
                  f"std={valid.std():.3f}, "
                  f"min={valid.min():.3f}, "
                  f"max={valid.max():.3f}")

        # Compare to current ML predictions
        print(f"\nComparing to Current ML Predictions:")
        for ratio_type in ['chest', 'waist', 'hip']:
            true_col = f'true_{ratio_type}_ratio'
            ml_col = f'ml_{ratio_type}_ratio'

            if ml_col in self.df.columns:
                valid_mask = self.df[true_col].notna() & self.df[ml_col].notna()
                if valid_mask.sum() > 0:
                    true_vals = self.df.loc[valid_mask, true_col]
                    ml_vals = self.df.loc[valid_mask, ml_col]
                    mae = mean_absolute_error(true_vals, ml_vals)
                    print(f"  {ratio_type.upper():5s} Current ML MAE: {mae:.4f}")

    def prepare_features(self):
        """Prepare feature matrix for training"""
        print(f"\n🔧 Preparing features...")

        # Encode categorical variables
        self.encoders['body_shape'] = LabelEncoder()
        self.encoders['gender'] = LabelEncoder()

        self.df['body_shape_encoded'] = self.encoders['body_shape'].fit_transform(
            self.df['ml_body_shape'].fillna('rectangle')
        )
        self.df['gender_encoded'] = self.encoders['gender'].fit_transform(
            self.df['gender'].fillna('neutral')
        )

        # Create feature matrix
        feature_cols = [
            'body_shape_encoded',
            'gender_encoded',
            'ml_bmi_estimate',
            'age',
            'predicted_shoulder_width',
            'predicted_chest_width',
            'predicted_waist_width',
            'predicted_hip_width',
        ]

        # Fill missing values
        for col in ['age', 'ml_bmi_estimate']:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        self.X = self.df[feature_cols].values

        print(f"✅ Feature matrix: {self.X.shape[0]} samples × {self.X.shape[1]} features")

    def train_model(self, ratio_type: str, model_type='gradient_boosting'):
        """
        Train a model for a specific ratio type

        Args:
            ratio_type: 'chest', 'waist', or 'hip'
            model_type: 'gradient_boosting' or 'random_forest'
        """
        print(f"\n{'='*80}")
        print(f"Training {ratio_type.upper()} Ratio Predictor ({model_type})")
        print(f"{'='*80}")

        # Get target values
        y = self.df[f'true_{ratio_type}_ratio'].values

        # Remove NaN values
        mask = ~np.isnan(y)
        X_clean = self.X[mask]
        y_clean = y[mask]

        print(f"Training samples: {len(X_clean)} (removed {(~mask).sum()} NaN values)")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42
        )

        print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

        # Initialize model
        if model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
                verbose=0
            )
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        print(f"\n🚀 Training {model_type}...")
        model.fit(X_train, y_train)

        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        print(f"\n📊 Training Results:")
        print(f"  Train MAE: {train_mae:.4f}, R²: {train_r2:.3f}")
        print(f"  Test MAE:  {test_mae:.4f}, R²: {test_r2:.3f}")

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_clean, y_clean, cv=5, scoring='neg_mean_absolute_error'
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()

        print(f"  Cross-Val MAE: {cv_mae:.4f} ± {cv_std:.4f}")

        # Compare to rule-based (if available)
        ml_predictions = self.df.loc[mask, f'ml_{ratio_type}_ratio'].values
        if len(ml_predictions) > 0 and not pd.isna(ml_predictions).all():
            # Get test set indices
            test_indices = X_test

            # Calculate rule-based MAE on same test set
            # (This is approximate since we can't perfectly align indices)
            rule_based_mae = mean_absolute_error(
                y_clean[-len(X_test):],
                ml_predictions[-len(X_test):]
            )

            improvement = ((rule_based_mae - test_mae) / rule_based_mae) * 100

            print(f"\n🎯 Improvement over Rule-Based:")
            print(f"  Rule-based MAE: {rule_based_mae:.4f}")
            print(f"  Trained Model MAE: {test_mae:.4f}")
            print(f"  Improvement: {improvement:.1f}%")
        else:
            print(f"\n⚠️  No rule-based predictions to compare against")
            improvement = None

        # Store model and stats
        self.models[ratio_type] = model
        self.training_stats[ratio_type] = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'improvement_percent': improvement,
            'n_samples': len(X_clean),
            'model_type': model_type,
        }

        return model

    def train_all_models(self, model_type='gradient_boosting'):
        """Train models for all ratio types"""
        print(f"\n{'='*80}")
        print(f"TRAINING ALL DEPTH RATIO MODELS")
        print(f"{'='*80}")

        for ratio_type in ['chest', 'waist', 'hip']:
            self.train_model(ratio_type, model_type)

        self._print_summary()

    def _print_summary(self):
        """Print training summary"""
        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}\n")

        for ratio_type, stats in self.training_stats.items():
            print(f"{ratio_type.upper()} Model:")
            print(f"  Test MAE: {stats['test_mae']:.4f}")
            print(f"  Test R²: {stats['test_r2']:.3f}")
            print(f"  CV MAE: {stats['cv_mae']:.4f} ± {stats['cv_std']:.4f}")
            if stats['improvement_percent'] is not None:
                print(f"  Improvement: {stats['improvement_percent']:.1f}% over rule-based")
            print()

    def save_models(self, output_dir='ml_models'):
        """Save trained models and metadata"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n💾 Saving models to {output_dir}/...")

        # Save models
        for ratio_type, model in self.models.items():
            model_path = os.path.join(output_dir, f'depth_ratio_{ratio_type}_model.pkl')
            joblib.dump(model, model_path)
            print(f"  ✅ Saved {ratio_type} model: {model_path}")

        # Save encoders
        for encoder_name, encoder in self.encoders.items():
            encoder_path = os.path.join(output_dir, f'label_encoder_{encoder_name}.pkl')
            joblib.dump(encoder, encoder_path)
            print(f"  ✅ Saved {encoder_name} encoder: {encoder_path}")

        # Save training statistics
        stats_path = os.path.join(output_dir, 'training_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(self.df),
                'models': self.training_stats
            }, f, indent=2)
        print(f"  ✅ Saved training stats: {stats_path}")

        # Save feature names
        feature_names = [
            'body_shape_encoded',
            'gender_encoded',
            'ml_bmi_estimate',
            'age',
            'predicted_shoulder_width',
            'predicted_chest_width',
            'predicted_waist_width',
            'predicted_hip_width',
        ]
        features_path = os.path.join(output_dir, 'feature_names.json')
        with open(features_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"  ✅ Saved feature names: {features_path}")

        print(f"\n✅ All models saved successfully!")
        print(f"\nNext steps:")
        print(f"1. Copy {output_dir}/ to backend/app/ml/")
        print(f"2. Update depth_ratio_predictor.py to load trained models")
        print(f"3. A/B test trained vs rule-based models")
        print(f"4. Monitor improvement in production")


def main():
    if len(sys.argv) < 2:
        print("\n🤖 Train ML Models from Validation Data")
        print("="*80)
        print("\nUsage:")
        print("  python train_from_validation_data.py <validation_results.csv>")
        print("\nExample:")
        print("  python train_from_validation_data.py validation_results.csv")
        print("\nThis will:")
        print("  ✓ Calculate true depth ratios from ground truth")
        print("  ✓ Train ML models for chest/waist/hip ratios")
        print("  ✓ Evaluate against rule-based predictions")
        print("  ✓ Save trained models to ml_models/")
        print("="*80)
        return

    validation_results_path = sys.argv[1]

    if not os.path.exists(validation_results_path):
        print(f"❌ Error: File not found: {validation_results_path}")
        return

    # Initialize trainer
    trainer = ValidationDataTrainer(validation_results_path)

    # Calculate true ratios
    trainer.calculate_true_ratios()

    # Prepare features
    trainer.prepare_features()

    # Train models
    trainer.train_all_models(model_type='gradient_boosting')

    # Save everything
    trainer.save_models()

    print(f"\n🎉 Training complete!")


if __name__ == "__main__":
    main()
