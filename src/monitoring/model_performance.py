"""
Real-time model performance tracking and monitoring.

Tracks performance metrics over time and provides alerting
when performance degrades below acceptable thresholds.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """
    Monitor and track model performance metrics over time.
    """
    
    def __init__(
        self,
        model_name: str,
        task_type: str = "classification",
        baseline_metrics: Optional[Dict[str, float]] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
        window_size: int = 1000
    ):
        """
        Initialize performance monitor.
        
        Args:
            model_name: Name of the model being monitored
            task_type: Type of task ('classification' or 'regression')
            baseline_metrics: Baseline performance metrics
            alert_thresholds: Threshold degradation for alerts (e.g., {'accuracy': 0.05})
            window_size: Number of recent predictions to track
        """
        self.model_name = model_name
        self.task_type = task_type.lower()
        self.baseline_metrics = baseline_metrics or {}
        self.alert_thresholds = alert_thresholds or {}
        self.window_size = window_size
        
        # Performance history
        self.metrics_history: List[Dict[str, Any]] = []
        self.predictions_buffer: List[Dict[str, Any]] = []
        
        # Validate task type
        if self.task_type not in ['classification', 'regression']:
            raise ValueError(
                f"Invalid task_type: {task_type}. "
                "Must be 'classification' or 'regression'"
            )
        
        logger.info(
            f"Initialized ModelPerformanceMonitor for {model_name} "
            f"({task_type})"
        )
    
    def track_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Track performance metrics for a batch of predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for classification)
            timestamp: Timestamp for metrics
            metadata: Additional metadata
        
        Returns:
            Dictionary of calculated metrics
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate metrics
        metrics = self.calculate_performance(y_true, y_pred, y_pred_proba)
        
        # Add to history
        record = {
            'timestamp': timestamp,
            'metrics': metrics,
            'sample_size': len(y_true),
            'metadata': metadata or {}
        }
        self.metrics_history.append(record)
        
        # Store predictions in buffer
        for i in range(len(y_true)):
            pred_record = {
                'timestamp': timestamp,
                'y_true': y_true[i],
                'y_pred': y_pred[i]
            }
            if y_pred_proba is not None:
                pred_record['y_pred_proba'] = y_pred_proba[i]
            
            self.predictions_buffer.append(pred_record)
        
        # Maintain window size
        if len(self.predictions_buffer) > self.window_size:
            self.predictions_buffer = self.predictions_buffer[-self.window_size:]
        
        # Check for alerts
        alerts = self._check_alerts(metrics)
        if alerts:
            logger.warning(
                f"Performance alerts for {self.model_name}: {alerts}"
            )
        
        logger.info(
            f"Tracked metrics for {self.model_name}: "
            f"{', '.join(f'{k}={v:.4f}' for k, v in metrics.items())}"
        )
        
        return metrics
    
    def calculate_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics based on task type.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for classification)
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        try:
            if self.task_type == 'classification':
                metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
                
                # Handle binary vs multiclass
                average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
                
                metrics['precision'] = float(
                    precision_score(y_true, y_pred, average=average, zero_division=0)
                )
                metrics['recall'] = float(
                    recall_score(y_true, y_pred, average=average, zero_division=0)
                )
                metrics['f1'] = float(
                    f1_score(y_true, y_pred, average=average, zero_division=0)
                )
                
                # ROC AUC if probabilities available
                if y_pred_proba is not None:
                    try:
                        if len(np.unique(y_true)) == 2:
                            # Binary classification
                            if y_pred_proba.ndim == 2:
                                proba = y_pred_proba[:, 1]
                            else:
                                proba = y_pred_proba
                            metrics['roc_auc'] = float(
                                roc_auc_score(y_true, proba)
                            )
                        else:
                            # Multiclass
                            metrics['roc_auc'] = float(
                                roc_auc_score(
                                    y_true, y_pred_proba,
                                    multi_class='ovr', average='weighted'
                                )
                            )
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC AUC: {e}")
            
            else:  # regression
                metrics['mse'] = float(mean_squared_error(y_true, y_pred))
                metrics['rmse'] = float(np.sqrt(metrics['mse']))
                metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
                metrics['r2'] = float(r2_score(y_true, y_pred))
                
                # Additional regression metrics
                metrics['mape'] = float(
                    np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
                )
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
        
        return metrics
    
    def compare_models(
        self,
        other_monitor: 'ModelPerformanceMonitor',
        metric: str = 'accuracy',
        window: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare performance with another model.
        
        Args:
            other_monitor: Another ModelPerformanceMonitor instance
            metric: Metric to compare
            window: Number of recent records to compare (None = all)
        
        Returns:
            Comparison results
        """
        # Get recent metrics
        self_metrics = self.get_recent_metrics(metric, window)
        other_metrics = other_monitor.get_recent_metrics(metric, window)
        
        if not self_metrics or not other_metrics:
            logger.warning("Insufficient data for comparison")
            return {
                'comparison_valid': False,
                'reason': 'Insufficient data'
            }
        
        self_mean = np.mean(self_metrics)
        other_mean = np.mean(other_metrics)
        
        # Statistical test (t-test)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(self_metrics, other_metrics)
        
        result = {
            'comparison_valid': True,
            'model_1': self.model_name,
            'model_2': other_monitor.model_name,
            'metric': metric,
            'model_1_mean': float(self_mean),
            'model_2_mean': float(other_mean),
            'difference': float(self_mean - other_mean),
            'percent_difference': float(
                (self_mean - other_mean) / other_mean * 100
            ) if other_mean != 0 else 0.0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significantly_different': p_value < 0.05,
            'better_model': (
                self.model_name if self_mean > other_mean
                else other_monitor.model_name
            )
        }
        
        logger.info(
            f"Model comparison: {self.model_name} vs {other_monitor.model_name} "
            f"on {metric}: {result['difference']:.4f} "
            f"({'significant' if result['significantly_different'] else 'not significant'})"
        )
        
        return result
    
    def get_recent_metrics(
        self,
        metric: str,
        window: Optional[int] = None
    ) -> List[float]:
        """
        Get recent values for a specific metric.
        
        Args:
            metric: Metric name
            window: Number of recent records (None = all)
        
        Returns:
            List of metric values
        """
        history = self.metrics_history
        if window:
            history = history[-window:]
        
        return [
            record['metrics'].get(metric)
            for record in history
            if metric in record['metrics']
        ]
    
    def _check_alerts(self, current_metrics: Dict[str, float]) -> List[str]:
        """
        Check if current metrics trigger any alerts.
        
        Args:
            current_metrics: Current metric values
        
        Returns:
            List of alert messages
        """
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if metric not in current_metrics:
                continue
            
            baseline = self.baseline_metrics.get(metric)
            if baseline is None:
                continue
            
            current = current_metrics[metric]
            
            # Check degradation
            if self.task_type == 'classification' or metric in ['r2']:
                # Higher is better
                degradation = baseline - current
                if degradation > threshold:
                    alerts.append(
                        f"{metric} degraded by {degradation:.4f} "
                        f"(current: {current:.4f}, baseline: {baseline:.4f})"
                    )
            else:
                # Lower is better (errors)
                degradation = current - baseline
                if degradation > threshold:
                    alerts.append(
                        f"{metric} increased by {degradation:.4f} "
                        f"(current: {current:.4f}, baseline: {baseline:.4f})"
                    )
        
        return alerts
    
    def get_performance_summary(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get summary of performance metrics.
        
        Args:
            time_window: Time window for summary (None = all time)
        
        Returns:
            Performance summary
        """
        if not self.metrics_history:
            return {
                'model_name': self.model_name,
                'total_predictions': 0,
                'metrics': {}
            }
        
        # Filter by time window
        history = self.metrics_history
        if time_window:
            cutoff = datetime.now() - time_window
            history = [
                r for r in history
                if r['timestamp'] >= cutoff
            ]
        
        # Aggregate metrics
        all_metrics = {}
        for record in history:
            for metric, value in record['metrics'].items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate statistics
        metrics_summary = {}
        for metric, values in all_metrics.items():
            metrics_summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'current': float(values[-1]) if values else None
            }
        
        total_predictions = sum(r['sample_size'] for r in history)
        
        return {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'total_predictions': total_predictions,
            'num_evaluations': len(history),
            'time_range': {
                'start': history[0]['timestamp'].isoformat(),
                'end': history[-1]['timestamp'].isoformat()
            } if history else None,
            'metrics': metrics_summary
        }
