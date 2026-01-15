"""
Prediction drift monitoring for detecting changes in model output distributions.

Monitors prediction distributions to detect concept drift and model degradation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


class PredictionDriftMonitor:
    """
    Monitor drift in model predictions over time.
    """
    
    def __init__(
        self,
        reference_predictions: np.ndarray,
        task_type: str = "classification",
        drift_threshold: float = 0.1,
        window_size: int = 1000,
        min_samples: int = 100
    ):
        """
        Initialize prediction drift monitor.
        
        Args:
            reference_predictions: Reference/baseline predictions
            task_type: Type of task ('classification' or 'regression')
            drift_threshold: Threshold for drift detection
            window_size: Size of sliding window for monitoring
            min_samples: Minimum samples needed for drift detection
        """
        self.reference_predictions = np.array(reference_predictions)
        self.task_type = task_type.lower()
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Prediction history
        self.predictions_history: List[Dict[str, Any]] = []
        self.drift_events: List[Dict[str, Any]] = []
        
        # Calculate reference statistics
        self._calculate_reference_stats()
        
        logger.info(
            f"Initialized PredictionDriftMonitor with "
            f"{len(reference_predictions)} reference predictions"
        )
    
    def _calculate_reference_stats(self) -> None:
        """Calculate statistics for reference predictions."""
        if self.task_type == 'classification':
            # Calculate class distribution
            unique, counts = np.unique(
                self.reference_predictions, return_counts=True
            )
            self.reference_distribution = dict(zip(unique, counts / len(counts)))
            
        else:  # regression
            self.reference_mean = float(np.mean(self.reference_predictions))
            self.reference_std = float(np.std(self.reference_predictions))
            self.reference_median = float(np.median(self.reference_predictions))
            self.reference_q25 = float(np.percentile(self.reference_predictions, 25))
            self.reference_q75 = float(np.percentile(self.reference_predictions, 75))
    
    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect drift in current predictions compared to reference.
        
        Args:
            current_predictions: Current model predictions
            timestamp: Timestamp for detection
            metadata: Additional metadata
        
        Returns:
            Drift detection results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        current_predictions = np.array(current_predictions)
        
        # Store predictions
        self.predictions_history.append({
            'timestamp': timestamp,
            'predictions': current_predictions,
            'metadata': metadata or {}
        })
        
        # Maintain window size
        if len(self.predictions_history) > self.window_size:
            self.predictions_history = self.predictions_history[-self.window_size:]
        
        # Check if we have enough samples
        if len(current_predictions) < self.min_samples:
            logger.warning(
                f"Insufficient samples for drift detection: "
                f"{len(current_predictions)} < {self.min_samples}"
            )
            return {
                'drift_detected': False,
                'reason': 'insufficient_samples',
                'sample_size': len(current_predictions)
            }
        
        # Detect drift based on task type
        if self.task_type == 'classification':
            result = self._detect_classification_drift(
                current_predictions, timestamp
            )
        else:
            result = self._detect_regression_drift(
                current_predictions, timestamp
            )
        
        # Log drift event if detected
        if result['drift_detected']:
            self.drift_events.append({
                'timestamp': timestamp,
                'result': result,
                'metadata': metadata
            })
            logger.warning(
                f"Prediction drift detected at {timestamp}: {result['drift_score']:.4f}"
            )
        
        return result
    
    def _detect_classification_drift(
        self,
        current_predictions: np.ndarray,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Detect drift for classification predictions.
        
        Args:
            current_predictions: Current predictions
            timestamp: Timestamp
        
        Returns:
            Drift detection results
        """
        # Calculate current distribution
        unique, counts = np.unique(current_predictions, return_counts=True)
        current_distribution = dict(zip(unique, counts / len(counts)))
        
        # Align distributions
        all_classes = set(self.reference_distribution.keys()) | set(
            current_distribution.keys()
        )
        
        ref_probs = [
            self.reference_distribution.get(cls, 0) for cls in all_classes
        ]
        cur_probs = [
            current_distribution.get(cls, 0) for cls in all_classes
        ]
        
        # Calculate Jensen-Shannon divergence
        js_divergence = jensenshannon(ref_probs, cur_probs)
        
        # Chi-square test
        ref_counts = np.array(ref_probs) * len(self.reference_predictions)
        cur_counts = np.array(cur_probs) * len(current_predictions)
        
        # Avoid division by zero
        ref_counts = np.where(ref_counts == 0, 0.5, ref_counts)
        
        chi2_stat = np.sum((cur_counts - ref_counts) ** 2 / ref_counts)
        df = len(all_classes) - 1
        p_value = 1 - stats.chi2.cdf(chi2_stat, df) if df > 0 else 1.0
        
        drift_detected = js_divergence > self.drift_threshold or p_value < 0.05
        
        return {
            'drift_detected': drift_detected,
            'drift_score': float(js_divergence),
            'method': 'jensen_shannon',
            'chi2_statistic': float(chi2_stat),
            'chi2_pvalue': float(p_value),
            'timestamp': timestamp.isoformat(),
            'sample_size': len(current_predictions),
            'reference_distribution': self.reference_distribution,
            'current_distribution': current_distribution
        }
    
    def _detect_regression_drift(
        self,
        current_predictions: np.ndarray,
        timestamp: datetime
    ) -> Dict[str, Any]:
        """
        Detect drift for regression predictions.
        
        Args:
            current_predictions: Current predictions
            timestamp: Timestamp
        
        Returns:
            Drift detection results
        """
        # Calculate current statistics
        current_mean = float(np.mean(current_predictions))
        current_std = float(np.std(current_predictions))
        current_median = float(np.median(current_predictions))
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.ks_2samp(
            self.reference_predictions, current_predictions
        )
        
        # T-test for mean
        t_stat, t_pvalue = stats.ttest_ind(
            self.reference_predictions, current_predictions
        )
        
        # Calculate relative changes
        mean_change = abs(current_mean - self.reference_mean) / (
            abs(self.reference_mean) + 1e-10
        )
        std_change = abs(current_std - self.reference_std) / (
            abs(self.reference_std) + 1e-10
        )
        
        # Drift detection based on multiple criteria
        drift_detected = (
            ks_pvalue < 0.05 or
            mean_change > self.drift_threshold or
            std_change > self.drift_threshold
        )
        
        # Combined drift score
        drift_score = float(ks_stat + mean_change + std_change) / 3
        
        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'method': 'ks_test',
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            't_statistic': float(t_stat),
            't_pvalue': float(t_pvalue),
            'timestamp': timestamp.isoformat(),
            'sample_size': len(current_predictions),
            'reference_stats': {
                'mean': self.reference_mean,
                'std': self.reference_std,
                'median': self.reference_median
            },
            'current_stats': {
                'mean': current_mean,
                'std': current_std,
                'median': current_median
            },
            'relative_changes': {
                'mean_change': float(mean_change),
                'std_change': float(std_change)
            }
        }
    
    def compare_distributions(
        self,
        predictions_1: np.ndarray,
        predictions_2: np.ndarray,
        method: str = "auto"
    ) -> Dict[str, Any]:
        """
        Compare two prediction distributions.
        
        Args:
            predictions_1: First set of predictions
            predictions_2: Second set of predictions
            method: Comparison method ('auto', 'ks', 'js', 'wasserstein')
        
        Returns:
            Comparison results
        """
        predictions_1 = np.array(predictions_1)
        predictions_2 = np.array(predictions_2)
        
        results = {}
        
        # Auto-select method based on task type
        if method == "auto":
            method = "js" if self.task_type == "classification" else "ks"
        
        if method == "ks" or self.task_type == "regression":
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(predictions_1, predictions_2)
            results['ks_statistic'] = float(ks_stat)
            results['ks_pvalue'] = float(ks_pvalue)
            results['method'] = 'ks_test'
            results['significantly_different'] = ks_pvalue < 0.05
        
        if method == "js" or self.task_type == "classification":
            # Jensen-Shannon divergence
            unique_1, counts_1 = np.unique(predictions_1, return_counts=True)
            unique_2, counts_2 = np.unique(predictions_2, return_counts=True)
            
            dist_1 = dict(zip(unique_1, counts_1 / len(counts_1)))
            dist_2 = dict(zip(unique_2, counts_2 / len(counts_2)))
            
            all_classes = set(dist_1.keys()) | set(dist_2.keys())
            probs_1 = [dist_1.get(cls, 0) for cls in all_classes]
            probs_2 = [dist_2.get(cls, 0) for cls in all_classes]
            
            js_div = jensenshannon(probs_1, probs_2)
            results['js_divergence'] = float(js_div)
            results['method'] = 'jensen_shannon'
            results['significantly_different'] = js_div > self.drift_threshold
        
        if method == "wasserstein":
            # Wasserstein distance
            from scipy.stats import wasserstein_distance
            wass_dist = wasserstein_distance(predictions_1, predictions_2)
            results['wasserstein_distance'] = float(wass_dist)
            results['method'] = 'wasserstein'
            results['significantly_different'] = wass_dist > self.drift_threshold
        
        return results
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """
        Get summary of drift events and monitoring status.
        
        Returns:
            Drift summary
        """
        if not self.predictions_history:
            return {
                'total_predictions': 0,
                'drift_events': 0,
                'drift_rate': 0.0
            }
        
        total_predictions = sum(
            len(record['predictions'])
            for record in self.predictions_history
        )
        
        return {
            'total_predictions': total_predictions,
            'monitoring_windows': len(self.predictions_history),
            'drift_events': len(self.drift_events),
            'drift_rate': len(self.drift_events) / len(self.predictions_history),
            'recent_drift_events': [
                {
                    'timestamp': event['timestamp'].isoformat(),
                    'drift_score': event['result']['drift_score']
                }
                for event in self.drift_events[-10:]
            ]
        }
