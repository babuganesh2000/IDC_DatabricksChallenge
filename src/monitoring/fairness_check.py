"""
Fairness and bias validation for ML models.

Implements various fairness metrics and bias detection methods.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FairnessChecker:
    """
    Check for bias and fairness issues in model predictions.
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        favorable_label: Any = 1,
        unfavorable_label: Any = 0,
        privileged_groups: Optional[Dict[str, List[Any]]] = None,
        unprivileged_groups: Optional[Dict[str, List[Any]]] = None
    ):
        """
        Initialize fairness checker.
        
        Args:
            protected_attributes: List of protected attribute names
            favorable_label: Label considered favorable (e.g., 1 for approved)
            unfavorable_label: Label considered unfavorable
            privileged_groups: Dict mapping attributes to privileged values
            unprivileged_groups: Dict mapping attributes to unprivileged values
        """
        self.protected_attributes = protected_attributes
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.privileged_groups = privileged_groups or {}
        self.unprivileged_groups = unprivileged_groups or {}
        
        # Fairness history
        self.fairness_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"Initialized FairnessChecker with protected attributes: "
            f"{', '.join(protected_attributes)}"
        )
    
    def check_demographic_parity(
        self,
        y_pred: np.ndarray,
        protected_attr: pd.Series,
        protected_attr_name: str
    ) -> Dict[str, Any]:
        """
        Check demographic parity (statistical parity).
        
        Measures whether the proportion of positive predictions is equal
        across different groups.
        
        Args:
            y_pred: Predicted labels
            protected_attr: Protected attribute values
            protected_attr_name: Name of protected attribute
        
        Returns:
            Demographic parity results
        """
        y_pred = np.array(y_pred)
        protected_attr = np.array(protected_attr)
        
        # Calculate selection rates for each group
        unique_groups = np.unique(protected_attr)
        selection_rates = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            group_preds = y_pred[mask]
            selection_rate = np.mean(group_preds == self.favorable_label)
            selection_rates[str(group)] = float(selection_rate)
        
        # Calculate disparate impact
        if len(selection_rates) >= 2:
            rates = list(selection_rates.values())
            disparate_impact = min(rates) / max(rates) if max(rates) > 0 else 0
        else:
            disparate_impact = 1.0
        
        # Calculate maximum difference
        max_diff = max(selection_rates.values()) - min(selection_rates.values())
        
        # Fairness threshold: 0.8 (80% rule)
        is_fair = disparate_impact >= 0.8
        
        result = {
            'metric': 'demographic_parity',
            'protected_attribute': protected_attr_name,
            'selection_rates': selection_rates,
            'disparate_impact': float(disparate_impact),
            'max_difference': float(max_diff),
            'is_fair': is_fair,
            'threshold': 0.8
        }
        
        if not is_fair:
            logger.warning(
                f"Demographic parity violation for {protected_attr_name}: "
                f"disparate impact = {disparate_impact:.3f}"
            )
        
        return result
    
    def check_equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: pd.Series,
        protected_attr_name: str
    ) -> Dict[str, Any]:
        """
        Check equal opportunity.
        
        Measures whether true positive rates are equal across groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attr: Protected attribute values
            protected_attr_name: Name of protected attribute
        
        Returns:
            Equal opportunity results
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        protected_attr = np.array(protected_attr)
        
        # Calculate TPR for each group
        unique_groups = np.unique(protected_attr)
        tpr_by_group = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            # True positives: actually positive and predicted positive
            positive_mask = group_true == self.favorable_label
            if positive_mask.sum() > 0:
                tpr = np.mean(
                    group_pred[positive_mask] == self.favorable_label
                )
            else:
                tpr = 0.0
            
            tpr_by_group[str(group)] = float(tpr)
        
        # Calculate difference in TPR
        if len(tpr_by_group) >= 2:
            max_tpr_diff = max(tpr_by_group.values()) - min(tpr_by_group.values())
        else:
            max_tpr_diff = 0.0
        
        # Fairness threshold: difference < 0.1
        is_fair = max_tpr_diff < 0.1
        
        result = {
            'metric': 'equal_opportunity',
            'protected_attribute': protected_attr_name,
            'tpr_by_group': tpr_by_group,
            'max_tpr_difference': float(max_tpr_diff),
            'is_fair': is_fair,
            'threshold': 0.1
        }
        
        if not is_fair:
            logger.warning(
                f"Equal opportunity violation for {protected_attr_name}: "
                f"max TPR difference = {max_tpr_diff:.3f}"
            )
        
        return result
    
    def calculate_disparate_impact(
        self,
        y_pred: np.ndarray,
        protected_attr: pd.Series,
        privileged_value: Any,
        unprivileged_value: Any
    ) -> float:
        """
        Calculate disparate impact ratio.
        
        Ratio of positive prediction rate for unprivileged group
        to privileged group. Should be >= 0.8 for fairness.
        
        Args:
            y_pred: Predicted labels
            protected_attr: Protected attribute values
            privileged_value: Value indicating privileged group
            unprivileged_value: Value indicating unprivileged group
        
        Returns:
            Disparate impact ratio
        """
        y_pred = np.array(y_pred)
        protected_attr = np.array(protected_attr)
        
        # Privileged group selection rate
        priv_mask = protected_attr == privileged_value
        priv_rate = np.mean(y_pred[priv_mask] == self.favorable_label)
        
        # Unprivileged group selection rate
        unpriv_mask = protected_attr == unprivileged_value
        unpriv_rate = np.mean(y_pred[unpriv_mask] == self.favorable_label)
        
        # Calculate ratio
        if priv_rate > 0:
            disparate_impact = unpriv_rate / priv_rate
        else:
            disparate_impact = 0.0
        
        return float(disparate_impact)
    
    def check_equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: pd.Series,
        protected_attr_name: str
    ) -> Dict[str, Any]:
        """
        Check equalized odds.
        
        Measures whether TPR and FPR are equal across groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attr: Protected attribute values
            protected_attr_name: Name of protected attribute
        
        Returns:
            Equalized odds results
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        protected_attr = np.array(protected_attr)
        
        unique_groups = np.unique(protected_attr)
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in unique_groups:
            mask = protected_attr == group
            group_true = y_true[mask]
            group_pred = y_pred[mask]
            
            # TPR
            positive_mask = group_true == self.favorable_label
            if positive_mask.sum() > 0:
                tpr = np.mean(
                    group_pred[positive_mask] == self.favorable_label
                )
            else:
                tpr = 0.0
            
            # FPR
            negative_mask = group_true == self.unfavorable_label
            if negative_mask.sum() > 0:
                fpr = np.mean(
                    group_pred[negative_mask] == self.favorable_label
                )
            else:
                fpr = 0.0
            
            tpr_by_group[str(group)] = float(tpr)
            fpr_by_group[str(group)] = float(fpr)
        
        # Calculate differences
        max_tpr_diff = (
            max(tpr_by_group.values()) - min(tpr_by_group.values())
            if tpr_by_group else 0.0
        )
        max_fpr_diff = (
            max(fpr_by_group.values()) - min(fpr_by_group.values())
            if fpr_by_group else 0.0
        )
        
        # Fairness threshold
        is_fair = max_tpr_diff < 0.1 and max_fpr_diff < 0.1
        
        result = {
            'metric': 'equalized_odds',
            'protected_attribute': protected_attr_name,
            'tpr_by_group': tpr_by_group,
            'fpr_by_group': fpr_by_group,
            'max_tpr_difference': float(max_tpr_diff),
            'max_fpr_difference': float(max_fpr_diff),
            'is_fair': is_fair,
            'threshold': 0.1
        }
        
        if not is_fair:
            logger.warning(
                f"Equalized odds violation for {protected_attr_name}"
            )
        
        return result
    
    def comprehensive_fairness_check(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        data: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run comprehensive fairness checks on all protected attributes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            data: DataFrame with protected attributes
        
        Returns:
            Dictionary of fairness results per attribute
        """
        results = {}
        
        for attr in self.protected_attributes:
            if attr not in data.columns:
                logger.warning(f"Protected attribute {attr} not in data")
                continue
            
            attr_results = {}
            
            # Demographic parity
            attr_results['demographic_parity'] = self.check_demographic_parity(
                y_pred, data[attr], attr
            )
            
            # Equal opportunity
            attr_results['equal_opportunity'] = self.check_equal_opportunity(
                y_true, y_pred, data[attr], attr
            )
            
            # Equalized odds
            attr_results['equalized_odds'] = self.check_equalized_odds(
                y_true, y_pred, data[attr], attr
            )
            
            # Overall fairness for this attribute
            all_fair = all(
                r['is_fair'] for r in attr_results.values()
            )
            attr_results['overall_fair'] = all_fair
            
            results[attr] = attr_results
        
        # Store in history
        self.fairness_history.append({
            'timestamp': pd.Timestamp.now(),
            'results': results
        })
        
        # Summary
        total_checks = len(results) * 3  # 3 metrics per attribute
        fair_checks = sum(
            sum(1 for r in attr_res.values() if isinstance(r, dict) and r.get('is_fair', False))
            for attr_res in results.values()
        )
        
        logger.info(
            f"Fairness check complete: {fair_checks}/{total_checks} "
            f"metrics passed fairness thresholds"
        )
        
        return results
    
    def generate_fairness_report(
        self,
        fairness_results: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Generate human-readable fairness report.
        
        Args:
            fairness_results: Results from comprehensive_fairness_check
        
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 70,
            "FAIRNESS AND BIAS REPORT",
            "=" * 70,
            ""
        ]
        
        for attr, results in fairness_results.items():
            report_lines.append(f"\nProtected Attribute: {attr}")
            report_lines.append("-" * 70)
            
            # Demographic parity
            dp = results.get('demographic_parity', {})
            if dp:
                report_lines.append(f"\n  Demographic Parity:")
                report_lines.append(
                    f"    Disparate Impact: {dp.get('disparate_impact', 0):.3f} "
                    f"({'PASS' if dp.get('is_fair') else 'FAIL'})"
                )
                report_lines.append(f"    Selection Rates:")
                for group, rate in dp.get('selection_rates', {}).items():
                    report_lines.append(f"      {group}: {rate:.3f}")
            
            # Equal opportunity
            eo = results.get('equal_opportunity', {})
            if eo:
                report_lines.append(f"\n  Equal Opportunity:")
                report_lines.append(
                    f"    Max TPR Difference: {eo.get('max_tpr_difference', 0):.3f} "
                    f"({'PASS' if eo.get('is_fair') else 'FAIL'})"
                )
                report_lines.append(f"    TPR by Group:")
                for group, tpr in eo.get('tpr_by_group', {}).items():
                    report_lines.append(f"      {group}: {tpr:.3f}")
            
            # Equalized odds
            eqo = results.get('equalized_odds', {})
            if eqo:
                report_lines.append(f"\n  Equalized Odds:")
                report_lines.append(
                    f"    Max TPR Difference: {eqo.get('max_tpr_difference', 0):.3f} "
                    f"({'PASS' if eqo.get('is_fair') else 'FAIL'})"
                )
                report_lines.append(
                    f"    Max FPR Difference: {eqo.get('max_fpr_difference', 0):.3f}"
                )
            
            # Overall
            overall = results.get('overall_fair', False)
            report_lines.append(
                f"\n  Overall: {'✓ FAIR' if overall else '✗ UNFAIR'}"
            )
        
        report_lines.append("\n" + "=" * 70)
        
        return "\n".join(report_lines)
