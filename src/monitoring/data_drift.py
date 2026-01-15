"""
Data drift detection using statistical methods.

Supports multiple drift detection methods:
- Kolmogorov-Smirnov test for numerical features
- Population Stability Index (PSI) for numerical features
- Wasserstein distance for numerical features
- Chi-square test for categorical features
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import wasserstein_distance

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detect statistical drift between reference and production data.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        ks_threshold: float = 0.05,
        psi_threshold: float = 0.2,
        wasserstein_threshold: float = 0.1,
        chi2_threshold: float = 0.05
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference/baseline dataset
            feature_columns: List of features to monitor
            categorical_features: List of categorical feature names
            ks_threshold: P-value threshold for KS test
            psi_threshold: PSI threshold for drift detection
            wasserstein_threshold: Wasserstein distance threshold
            chi2_threshold: P-value threshold for Chi-square test
        """
        self.reference_data = reference_data
        self.feature_columns = feature_columns or list(reference_data.columns)
        self.categorical_features = categorical_features or []
        self.numerical_features = [
            f for f in self.feature_columns 
            if f not in self.categorical_features
        ]
        
        self.ks_threshold = ks_threshold
        self.psi_threshold = psi_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.chi2_threshold = chi2_threshold
        
        self._validate_data()
        logger.info(
            f"Initialized DataDriftDetector with {len(self.numerical_features)} "
            f"numerical and {len(self.categorical_features)} categorical features"
        )
    
    def _validate_data(self) -> None:
        """Validate reference data."""
        missing_features = [
            f for f in self.feature_columns 
            if f not in self.reference_data.columns
        ]
        if missing_features:
            raise ValueError(
                f"Missing features in reference data: {missing_features}"
            )
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Union[float, bool, str]]]:
        """
        Detect drift using multiple statistical methods.
        
        Args:
            current_data: Current production data
            methods: List of methods to use ['ks', 'psi', 'wasserstein', 'chi2']
                    If None, uses all applicable methods
        
        Returns:
            Dictionary with drift detection results per feature
        """
        if methods is None:
            methods = ['ks', 'psi', 'wasserstein', 'chi2']
        
        results = {}
        
        # Numerical features
        for feature in self.numerical_features:
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} not in current data")
                continue
            
            ref_values = self.reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                logger.warning(f"Empty values for feature {feature}")
                continue
            
            feature_results = {
                'feature_type': 'numerical',
                'drift_detected': False
            }
            
            # KS test
            if 'ks' in methods:
                ks_stat, ks_pvalue = self.calculate_ks_statistic(
                    ref_values, cur_values
                )
                feature_results['ks_statistic'] = float(ks_stat)
                feature_results['ks_pvalue'] = float(ks_pvalue)
                feature_results['ks_drift'] = ks_pvalue < self.ks_threshold
                if feature_results['ks_drift']:
                    feature_results['drift_detected'] = True
            
            # PSI
            if 'psi' in methods:
                psi_value = self.calculate_psi(ref_values, cur_values)
                feature_results['psi'] = float(psi_value)
                feature_results['psi_drift'] = psi_value > self.psi_threshold
                if feature_results['psi_drift']:
                    feature_results['drift_detected'] = True
            
            # Wasserstein distance
            if 'wasserstein' in methods:
                wass_dist = self.calculate_wasserstein_distance(
                    ref_values, cur_values
                )
                feature_results['wasserstein_distance'] = float(wass_dist)
                feature_results['wasserstein_drift'] = (
                    wass_dist > self.wasserstein_threshold
                )
                if feature_results['wasserstein_drift']:
                    feature_results['drift_detected'] = True
            
            results[feature] = feature_results
        
        # Categorical features
        if 'chi2' in methods:
            for feature in self.categorical_features:
                if feature not in current_data.columns:
                    logger.warning(f"Feature {feature} not in current data")
                    continue
                
                ref_values = self.reference_data[feature].dropna()
                cur_values = current_data[feature].dropna()
                
                if len(ref_values) == 0 or len(cur_values) == 0:
                    logger.warning(f"Empty values for feature {feature}")
                    continue
                
                chi2_stat, chi2_pvalue = self._calculate_chi2(
                    ref_values, cur_values
                )
                
                results[feature] = {
                    'feature_type': 'categorical',
                    'chi2_statistic': float(chi2_stat),
                    'chi2_pvalue': float(chi2_pvalue),
                    'chi2_drift': chi2_pvalue < self.chi2_threshold,
                    'drift_detected': chi2_pvalue < self.chi2_threshold
                }
        
        logger.info(
            f"Drift detection complete. Features with drift: "
            f"{sum(1 for r in results.values() if r['drift_detected'])}/{len(results)}"
        )
        
        return results
    
    def calculate_ks_statistic(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic.
        
        Args:
            reference: Reference distribution
            current: Current distribution
        
        Returns:
            Tuple of (KS statistic, p-value)
        """
        try:
            ks_stat, p_value = stats.ks_2samp(reference, current)
            return ks_stat, p_value
        except Exception as e:
            logger.error(f"Error calculating KS statistic: {e}")
            return 0.0, 1.0
    
    def calculate_psi(
        self,
        reference: pd.Series,
        current: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in distribution between two samples.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Small change
        PSI >= 0.2: Significant change
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization
        
        Returns:
            PSI value
        """
        try:
            # Create bins based on reference data
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            
            if min_val == max_val:
                return 0.0
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate distributions
            ref_hist, _ = np.histogram(reference, bins=bin_edges)
            cur_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to get percentages
            ref_pct = ref_hist / len(reference)
            cur_pct = cur_hist / len(current)
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
            
            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            
            return float(psi)
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def calculate_wasserstein_distance(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> float:
        """
        Calculate Wasserstein distance (Earth Mover's Distance).
        
        Args:
            reference: Reference distribution
            current: Current distribution
        
        Returns:
            Wasserstein distance
        """
        try:
            distance = wasserstein_distance(reference, current)
            return float(distance)
        except Exception as e:
            logger.error(f"Error calculating Wasserstein distance: {e}")
            return 0.0
    
    def _calculate_chi2(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Chi-square test for categorical features.
        
        Args:
            reference: Reference categorical values
            current: Current categorical values
        
        Returns:
            Tuple of (Chi-square statistic, p-value)
        """
        try:
            # Get all unique categories
            all_categories = set(reference.unique()) | set(current.unique())
            
            # Create frequency tables
            ref_counts = reference.value_counts()
            cur_counts = current.value_counts()
            
            # Align categories
            ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
            cur_freq = [cur_counts.get(cat, 0) for cat in all_categories]
            
            # Normalize to expected frequencies
            ref_freq = np.array(ref_freq) / sum(ref_freq) * sum(cur_freq)
            cur_freq = np.array(cur_freq)
            
            # Avoid division by zero
            ref_freq = np.where(ref_freq == 0, 0.5, ref_freq)
            
            # Chi-square test
            chi2_stat = np.sum((cur_freq - ref_freq) ** 2 / ref_freq)
            df = len(all_categories) - 1
            p_value = 1 - stats.chi2.cdf(chi2_stat, df)
            
            return chi2_stat, p_value
        except Exception as e:
            logger.error(f"Error calculating Chi-square: {e}")
            return 0.0, 1.0
    
    def generate_drift_report(
        self,
        drift_results: Dict[str, Dict[str, Union[float, bool, str]]]
    ) -> str:
        """
        Generate human-readable drift report.
        
        Args:
            drift_results: Results from detect_drift()
        
        Returns:
            Formatted report string
        """
        report_lines = ["=" * 60, "DATA DRIFT REPORT", "=" * 60, ""]
        
        drifted_features = [
            f for f, r in drift_results.items() if r.get('drift_detected', False)
        ]
        
        report_lines.append(
            f"Summary: {len(drifted_features)}/{len(drift_results)} "
            f"features show drift"
        )
        report_lines.append("")
        
        if drifted_features:
            report_lines.append("Features with detected drift:")
            report_lines.append("-" * 60)
            
            for feature in drifted_features:
                result = drift_results[feature]
                report_lines.append(f"\n{feature} ({result['feature_type']}):")
                
                if 'ks_pvalue' in result and result.get('ks_drift'):
                    report_lines.append(
                        f"  - KS Test: p-value={result['ks_pvalue']:.4f} "
                        f"(threshold={self.ks_threshold})"
                    )
                
                if 'psi' in result and result.get('psi_drift'):
                    report_lines.append(
                        f"  - PSI: {result['psi']:.4f} "
                        f"(threshold={self.psi_threshold})"
                    )
                
                if 'wasserstein_distance' in result and result.get('wasserstein_drift'):
                    report_lines.append(
                        f"  - Wasserstein: {result['wasserstein_distance']:.4f} "
                        f"(threshold={self.wasserstein_threshold})"
                    )
                
                if 'chi2_pvalue' in result and result.get('chi2_drift'):
                    report_lines.append(
                        f"  - Chi-square: p-value={result['chi2_pvalue']:.4f} "
                        f"(threshold={self.chi2_threshold})"
                    )
        else:
            report_lines.append("No drift detected in any features.")
        
        report_lines.append("\n" + "=" * 60)
        
        return "\n".join(report_lines)
