"""
A/B testing framework for comparing model variants.

Implements champion vs. challenger testing with statistical significance.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """A/B test status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


class ABTestManager:
    """
    Manage A/B tests for model comparison (Champion vs. Challenger).
    """
    
    def __init__(
        self,
        significance_level: float = 0.05,
        minimum_sample_size: int = 100,
        use_multi_armed_bandit: bool = False,
        exploration_rate: float = 0.1
    ):
        """
        Initialize A/B test manager.
        
        Args:
            significance_level: P-value threshold for statistical significance
            minimum_sample_size: Minimum samples per variant before testing
            use_multi_armed_bandit: Use MAB for adaptive allocation
            exploration_rate: Exploration rate for MAB (epsilon-greedy)
        """
        self.significance_level = significance_level
        self.minimum_sample_size = minimum_sample_size
        self.use_multi_armed_bandit = use_multi_armed_bandit
        self.exploration_rate = exploration_rate
        
        # Test registry
        self.tests: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"Initialized ABTestManager (MAB: {use_multi_armed_bandit})"
        )
    
    def create_test(
        self,
        test_name: str,
        champion_model: str,
        challenger_model: str,
        metric: str = "accuracy",
        higher_is_better: bool = True,
        traffic_split: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new A/B test.
        
        Args:
            test_name: Name of the test
            champion_model: Name/ID of champion model
            challenger_model: Name/ID of challenger model
            metric: Metric to compare
            higher_is_better: Whether higher metric values are better
            traffic_split: Traffic allocation {'champion': 0.5, 'challenger': 0.5}
            metadata: Additional test metadata
        
        Returns:
            Test ID
        """
        test_id = f"test_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if traffic_split is None:
            traffic_split = {'champion': 0.5, 'challenger': 0.5}
        
        test = {
            'id': test_id,
            'name': test_name,
            'champion': champion_model,
            'challenger': challenger_model,
            'metric': metric,
            'higher_is_better': higher_is_better,
            'traffic_split': traffic_split,
            'status': TestStatus.RUNNING,
            'created_at': datetime.now(),
            'completed_at': None,
            'results': {
                'champion': [],
                'challenger': []
            },
            'winner': None,
            'metadata': metadata or {}
        }
        
        self.tests[test_id] = test
        
        logger.info(
            f"Created A/B test: {test_name} "
            f"({champion_model} vs {challenger_model})"
        )
        
        return test_id
    
    def record_outcome(
        self,
        test_id: str,
        variant: str,
        outcome: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record an outcome for a test variant.
        
        Args:
            test_id: Test identifier
            variant: Variant name ('champion' or 'challenger')
            outcome: Metric value
            metadata: Additional outcome metadata
        
        Returns:
            Success status
        """
        if test_id not in self.tests:
            logger.error(f"Test not found: {test_id}")
            return False
        
        test = self.tests[test_id]
        
        if test['status'] != TestStatus.RUNNING:
            logger.warning(f"Test {test_id} is not running")
            return False
        
        if variant not in ['champion', 'challenger']:
            logger.error(f"Invalid variant: {variant}")
            return False
        
        # Record outcome
        test['results'][variant].append({
            'timestamp': datetime.now(),
            'outcome': outcome,
            'metadata': metadata or {}
        })
        
        # Update traffic allocation if using MAB
        if self.use_multi_armed_bandit:
            self._update_traffic_allocation(test_id)
        
        return True
    
    def _update_traffic_allocation(self, test_id: str) -> None:
        """
        Update traffic allocation using multi-armed bandit.
        
        Args:
            test_id: Test identifier
        """
        test = self.tests[test_id]
        
        champion_outcomes = [
            r['outcome'] for r in test['results']['champion']
        ]
        challenger_outcomes = [
            r['outcome'] for r in test['results']['challenger']
        ]
        
        if not champion_outcomes or not challenger_outcomes:
            return
        
        # Calculate current means
        champion_mean = np.mean(champion_outcomes)
        challenger_mean = np.mean(challenger_outcomes)
        
        # Epsilon-greedy allocation
        if test['higher_is_better']:
            better_variant = (
                'challenger' if challenger_mean > champion_mean
                else 'champion'
            )
        else:
            better_variant = (
                'challenger' if challenger_mean < champion_mean
                else 'champion'
            )
        
        # Update allocation
        test['traffic_split'] = {
            better_variant: 1.0 - self.exploration_rate,
            'champion' if better_variant == 'challenger' else 'challenger': (
                self.exploration_rate
            )
        }
        
        logger.info(
            f"Updated traffic allocation for {test_id}: {test['traffic_split']}"
        )
    
    def analyze_results(
        self,
        test_id: str
    ) -> Dict[str, Any]:
        """
        Analyze test results and determine statistical significance.
        
        Args:
            test_id: Test identifier
        
        Returns:
            Analysis results
        """
        if test_id not in self.tests:
            logger.error(f"Test not found: {test_id}")
            return {}
        
        test = self.tests[test_id]
        
        champion_outcomes = [
            r['outcome'] for r in test['results']['champion']
        ]
        challenger_outcomes = [
            r['outcome'] for r in test['results']['challenger']
        ]
        
        # Check minimum sample size
        if (len(champion_outcomes) < self.minimum_sample_size or
            len(challenger_outcomes) < self.minimum_sample_size):
            logger.warning(
                f"Insufficient samples for test {test_id}: "
                f"champion={len(champion_outcomes)}, "
                f"challenger={len(challenger_outcomes)}"
            )
            return {
                'test_id': test_id,
                'sufficient_data': False,
                'champion_samples': len(champion_outcomes),
                'challenger_samples': len(challenger_outcomes),
                'minimum_required': self.minimum_sample_size
            }
        
        # Calculate statistics
        champion_mean = np.mean(champion_outcomes)
        champion_std = np.std(champion_outcomes, ddof=1)
        challenger_mean = np.mean(challenger_outcomes)
        challenger_std = np.std(challenger_outcomes, ddof=1)
        
        # Statistical test (t-test)
        t_stat, p_value = stats.ttest_ind(
            challenger_outcomes, champion_outcomes
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(champion_outcomes) - 1) * champion_std ** 2 +
             (len(challenger_outcomes) - 1) * challenger_std ** 2) /
            (len(champion_outcomes) + len(challenger_outcomes) - 2)
        )
        
        cohens_d = (
            (challenger_mean - champion_mean) / pooled_std
            if pooled_std > 0 else 0
        )
        
        # Determine significance
        is_significant = p_value < self.significance_level
        
        # Determine winner
        if is_significant:
            if test['higher_is_better']:
                winner = (
                    'challenger' if challenger_mean > champion_mean
                    else 'champion'
                )
            else:
                winner = (
                    'challenger' if challenger_mean < champion_mean
                    else 'champion'
                )
        else:
            winner = 'no_significant_difference'
        
        # Calculate confidence intervals
        champion_ci = stats.t.interval(
            1 - self.significance_level,
            len(champion_outcomes) - 1,
            loc=champion_mean,
            scale=stats.sem(champion_outcomes)
        )
        
        challenger_ci = stats.t.interval(
            1 - self.significance_level,
            len(challenger_outcomes) - 1,
            loc=challenger_mean,
            scale=stats.sem(challenger_outcomes)
        )
        
        results = {
            'test_id': test_id,
            'test_name': test['name'],
            'metric': test['metric'],
            'sufficient_data': True,
            'champion': {
                'model': test['champion'],
                'samples': len(champion_outcomes),
                'mean': float(champion_mean),
                'std': float(champion_std),
                'confidence_interval': (float(champion_ci[0]), float(champion_ci[1]))
            },
            'challenger': {
                'model': test['challenger'],
                'samples': len(challenger_outcomes),
                'mean': float(challenger_mean),
                'std': float(challenger_std),
                'confidence_interval': (float(challenger_ci[0]), float(challenger_ci[1]))
            },
            'statistical_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significance_level': self.significance_level,
                'is_significant': is_significant,
                'cohens_d': float(cohens_d)
            },
            'winner': winner,
            'improvement': {
                'absolute': float(challenger_mean - champion_mean),
                'relative': float(
                    (challenger_mean - champion_mean) / champion_mean * 100
                    if champion_mean != 0 else 0
                )
            }
        }
        
        logger.info(
            f"Test analysis complete for {test_id}: "
            f"winner={winner}, p_value={p_value:.4f}"
        )
        
        return results
    
    def get_winner(
        self,
        test_id: str,
        auto_complete: bool = True
    ) -> Optional[str]:
        """
        Get the winner of a test.
        
        Args:
            test_id: Test identifier
            auto_complete: Automatically complete test if winner determined
        
        Returns:
            Winner variant name or None
        """
        analysis = self.analyze_results(test_id)
        
        if not analysis.get('sufficient_data'):
            logger.info(f"Test {test_id} has insufficient data")
            return None
        
        winner = analysis.get('winner')
        
        if winner and winner != 'no_significant_difference' and auto_complete:
            self.complete_test(test_id, winner)
        
        return winner
    
    def complete_test(
        self,
        test_id: str,
        winner: Optional[str] = None
    ) -> bool:
        """
        Complete a test.
        
        Args:
            test_id: Test identifier
            winner: Winner variant (if known)
        
        Returns:
            Success status
        """
        if test_id not in self.tests:
            logger.error(f"Test not found: {test_id}")
            return False
        
        test = self.tests[test_id]
        test['status'] = TestStatus.COMPLETED
        test['completed_at'] = datetime.now()
        
        if winner:
            test['winner'] = winner
        
        logger.info(f"Completed test {test_id}: winner={winner}")
        
        return True
    
    def stop_test(self, test_id: str) -> bool:
        """
        Stop a running test.
        
        Args:
            test_id: Test identifier
        
        Returns:
            Success status
        """
        if test_id not in self.tests:
            logger.error(f"Test not found: {test_id}")
            return False
        
        test = self.tests[test_id]
        test['status'] = TestStatus.STOPPED
        
        logger.info(f"Stopped test {test_id}")
        
        return True
    
    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """
        Get summary of a test.
        
        Args:
            test_id: Test identifier
        
        Returns:
            Test summary
        """
        if test_id not in self.tests:
            logger.error(f"Test not found: {test_id}")
            return {}
        
        test = self.tests[test_id]
        
        champion_count = len(test['results']['champion'])
        challenger_count = len(test['results']['challenger'])
        
        return {
            'test_id': test_id,
            'name': test['name'],
            'status': test['status'].value,
            'champion': test['champion'],
            'challenger': test['challenger'],
            'metric': test['metric'],
            'created_at': test['created_at'].isoformat(),
            'completed_at': (
                test['completed_at'].isoformat()
                if test['completed_at'] else None
            ),
            'samples': {
                'champion': champion_count,
                'challenger': challenger_count,
                'total': champion_count + challenger_count
            },
            'traffic_split': test['traffic_split'],
            'winner': test['winner']
        }
    
    def list_tests(
        self,
        status_filter: Optional[TestStatus] = None
    ) -> List[Dict[str, Any]]:
        """
        List all tests.
        
        Args:
            status_filter: Filter by status
        
        Returns:
            List of test summaries
        """
        tests = list(self.tests.values())
        
        if status_filter:
            tests = [t for t in tests if t['status'] == status_filter]
        
        return [self.get_test_summary(t['id']) for t in tests]
