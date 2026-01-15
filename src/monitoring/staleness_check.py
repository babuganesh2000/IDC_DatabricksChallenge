"""
Model staleness checking and freshness validation.

Monitors model age and data freshness to determine when retraining is needed.
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class ModelnessChecker:
    """
    Check model age and data freshness to determine retraining needs.
    """
    
    def __init__(
        self,
        model_name: str,
        training_date: datetime,
        max_model_age_days: int = 90,
        max_data_age_days: int = 7,
        performance_degradation_threshold: float = 0.05,
        drift_threshold: float = 0.2
    ):
        """
        Initialize staleness checker.
        
        Args:
            model_name: Name of the model
            training_date: Date when model was trained
            max_model_age_days: Maximum acceptable model age in days
            max_data_age_days: Maximum acceptable data age in days
            performance_degradation_threshold: Max acceptable performance drop
            drift_threshold: Threshold for data drift
        """
        self.model_name = model_name
        self.training_date = training_date
        self.max_model_age_days = max_model_age_days
        self.max_data_age_days = max_data_age_days
        self.performance_degradation_threshold = performance_degradation_threshold
        self.drift_threshold = drift_threshold
        
        # Tracking
        self.last_performance_check: Optional[datetime] = None
        self.last_drift_check: Optional[datetime] = None
        self.performance_history: list = []
        self.drift_history: list = []
        
        logger.info(
            f"Initialized ModelnessChecker for {model_name}, "
            f"trained on {training_date.isoformat()}"
        )
    
    def check_model_age(
        self,
        current_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Check if model is too old.
        
        Args:
            current_date: Current date (defaults to now)
        
        Returns:
            Model age check results
        """
        if current_date is None:
            current_date = datetime.now()
        
        age = current_date - self.training_date
        age_days = age.days
        
        is_stale = age_days > self.max_model_age_days
        
        # Calculate staleness score (0-1, where 1 is very stale)
        staleness_score = min(age_days / self.max_model_age_days, 2.0)
        
        result = {
            'model_name': self.model_name,
            'training_date': self.training_date.isoformat(),
            'current_date': current_date.isoformat(),
            'model_age_days': age_days,
            'max_age_days': self.max_model_age_days,
            'is_stale': is_stale,
            'staleness_score': float(staleness_score),
            'recommendation': (
                'Model retraining recommended' if is_stale
                else 'Model age is acceptable'
            )
        }
        
        if is_stale:
            logger.warning(
                f"Model {self.model_name} is stale: "
                f"{age_days} days old (max: {self.max_model_age_days})"
            )
        else:
            logger.info(
                f"Model {self.model_name} age is acceptable: {age_days} days"
            )
        
        return result
    
    def check_data_freshness(
        self,
        last_data_update: datetime,
        current_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Check if training data is too old.
        
        Args:
            last_data_update: Date of last data update
            current_date: Current date (defaults to now)
        
        Returns:
            Data freshness check results
        """
        if current_date is None:
            current_date = datetime.now()
        
        age = current_date - last_data_update
        age_days = age.days
        
        is_stale = age_days > self.max_data_age_days
        
        # Calculate staleness score
        staleness_score = min(age_days / self.max_data_age_days, 2.0)
        
        result = {
            'model_name': self.model_name,
            'last_data_update': last_data_update.isoformat(),
            'current_date': current_date.isoformat(),
            'data_age_days': age_days,
            'max_age_days': self.max_data_age_days,
            'is_stale': is_stale,
            'staleness_score': float(staleness_score),
            'recommendation': (
                'Data refresh recommended' if is_stale
                else 'Data freshness is acceptable'
            )
        }
        
        if is_stale:
            logger.warning(
                f"Data for {self.model_name} is stale: "
                f"{age_days} days old (max: {self.max_data_age_days})"
            )
        
        return result
    
    def check_performance_degradation(
        self,
        baseline_performance: float,
        current_performance: float,
        metric_name: str = "accuracy",
        higher_is_better: bool = True
    ) -> Dict[str, Any]:
        """
        Check if performance has degraded significantly.
        
        Args:
            baseline_performance: Baseline/training performance
            current_performance: Current production performance
            metric_name: Name of the metric
            higher_is_better: Whether higher values are better
        
        Returns:
            Performance degradation check results
        """
        # Calculate degradation
        if higher_is_better:
            degradation = baseline_performance - current_performance
        else:
            degradation = current_performance - baseline_performance
        
        relative_degradation = (
            degradation / abs(baseline_performance)
            if baseline_performance != 0 else 0
        )
        
        is_degraded = degradation > self.performance_degradation_threshold
        
        # Store in history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'baseline': baseline_performance,
            'current': current_performance,
            'degradation': degradation,
            'is_degraded': is_degraded
        })
        self.last_performance_check = datetime.now()
        
        result = {
            'model_name': self.model_name,
            'metric': metric_name,
            'baseline_performance': float(baseline_performance),
            'current_performance': float(current_performance),
            'absolute_degradation': float(degradation),
            'relative_degradation': float(relative_degradation),
            'threshold': self.performance_degradation_threshold,
            'is_degraded': is_degraded,
            'recommendation': (
                'Model retraining recommended due to performance degradation'
                if is_degraded else 'Performance is acceptable'
            )
        }
        
        if is_degraded:
            logger.warning(
                f"Performance degradation detected for {self.model_name}: "
                f"{metric_name} dropped by {degradation:.4f} "
                f"({baseline_performance:.4f} -> {current_performance:.4f})"
            )
        
        return result
    
    def check_drift_impact(
        self,
        drift_score: float,
        drift_method: str = "PSI"
    ) -> Dict[str, Any]:
        """
        Check if data drift requires retraining.
        
        Args:
            drift_score: Calculated drift score
            drift_method: Method used to calculate drift
        
        Returns:
            Drift impact check results
        """
        requires_retraining = drift_score > self.drift_threshold
        
        # Store in history
        self.drift_history.append({
            'timestamp': datetime.now(),
            'drift_score': drift_score,
            'method': drift_method,
            'requires_retraining': requires_retraining
        })
        self.last_drift_check = datetime.now()
        
        result = {
            'model_name': self.model_name,
            'drift_method': drift_method,
            'drift_score': float(drift_score),
            'threshold': self.drift_threshold,
            'requires_retraining': requires_retraining,
            'recommendation': (
                'Model retraining recommended due to data drift'
                if requires_retraining else 'Drift is within acceptable range'
            )
        }
        
        if requires_retraining:
            logger.warning(
                f"Significant drift detected for {self.model_name}: "
                f"{drift_method} score = {drift_score:.4f} "
                f"(threshold: {self.drift_threshold})"
            )
        
        return result
    
    def should_retrain(
        self,
        current_date: Optional[datetime] = None,
        last_data_update: Optional[datetime] = None,
        current_performance: Optional[float] = None,
        baseline_performance: Optional[float] = None,
        drift_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive check to determine if model should be retrained.
        
        Args:
            current_date: Current date
            last_data_update: Date of last data update
            current_performance: Current performance metric
            baseline_performance: Baseline performance metric
            drift_score: Data drift score
        
        Returns:
            Comprehensive retraining recommendation
        """
        if current_date is None:
            current_date = datetime.now()
        
        reasons = []
        checks = {}
        
        # Check model age
        age_check = self.check_model_age(current_date)
        checks['model_age'] = age_check
        if age_check['is_stale']:
            reasons.append(
                f"Model age ({age_check['model_age_days']} days) "
                f"exceeds threshold ({self.max_model_age_days} days)"
            )
        
        # Check data freshness
        if last_data_update is not None:
            freshness_check = self.check_data_freshness(
                last_data_update, current_date
            )
            checks['data_freshness'] = freshness_check
            if freshness_check['is_stale']:
                reasons.append(
                    f"Data age ({freshness_check['data_age_days']} days) "
                    f"exceeds threshold ({self.max_data_age_days} days)"
                )
        
        # Check performance degradation
        if current_performance is not None and baseline_performance is not None:
            perf_check = self.check_performance_degradation(
                baseline_performance, current_performance
            )
            checks['performance'] = perf_check
            if perf_check['is_degraded']:
                reasons.append(
                    f"Performance degraded by "
                    f"{perf_check['absolute_degradation']:.4f}"
                )
        
        # Check drift impact
        if drift_score is not None:
            drift_check = self.check_drift_impact(drift_score)
            checks['drift'] = drift_check
            if drift_check['requires_retraining']:
                reasons.append(
                    f"Data drift score ({drift_score:.4f}) "
                    f"exceeds threshold ({self.drift_threshold})"
                )
        
        should_retrain = len(reasons) > 0
        
        # Calculate overall urgency score (0-1)
        urgency_scores = []
        if 'model_age' in checks:
            urgency_scores.append(checks['model_age']['staleness_score'])
        if 'data_freshness' in checks:
            urgency_scores.append(checks['data_freshness']['staleness_score'])
        if 'performance' in checks and checks['performance']['is_degraded']:
            urgency_scores.append(
                checks['performance']['relative_degradation'] /
                self.performance_degradation_threshold
            )
        if 'drift' in checks and checks['drift']['requires_retraining']:
            urgency_scores.append(
                checks['drift']['drift_score'] / self.drift_threshold
            )
        
        urgency_score = (
            sum(urgency_scores) / len(urgency_scores)
            if urgency_scores else 0.0
        )
        
        result = {
            'model_name': self.model_name,
            'should_retrain': should_retrain,
            'urgency_score': float(min(urgency_score, 1.0)),
            'urgency_level': (
                'critical' if urgency_score > 1.5
                else 'high' if urgency_score > 1.0
                else 'medium' if urgency_score > 0.5
                else 'low'
            ),
            'reasons': reasons,
            'checks': checks,
            'timestamp': current_date.isoformat()
        }
        
        if should_retrain:
            logger.warning(
                f"Retraining recommended for {self.model_name}. "
                f"Urgency: {result['urgency_level']}. "
                f"Reasons: {', '.join(reasons)}"
            )
        else:
            logger.info(
                f"No retraining needed for {self.model_name} at this time"
            )
        
        return result
    
    def get_staleness_summary(self) -> Dict[str, Any]:
        """
        Get summary of staleness checks.
        
        Returns:
            Staleness summary
        """
        return {
            'model_name': self.model_name,
            'training_date': self.training_date.isoformat(),
            'thresholds': {
                'max_model_age_days': self.max_model_age_days,
                'max_data_age_days': self.max_data_age_days,
                'performance_degradation': self.performance_degradation_threshold,
                'drift_threshold': self.drift_threshold
            },
            'checks_performed': {
                'performance_checks': len(self.performance_history),
                'drift_checks': len(self.drift_history),
                'last_performance_check': (
                    self.last_performance_check.isoformat()
                    if self.last_performance_check else None
                ),
                'last_drift_check': (
                    self.last_drift_check.isoformat()
                    if self.last_drift_check else None
                )
            }
        }
