"""Automated model promotion with validation and approval workflows.

Handles model evaluation, comparison, and promotion across stages.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.tracking import MlflowClient

from ..utils.logging_utils import get_logger
from .model_registry import ModelRegistryManager

logger = get_logger(__name__)


class ModelPromoter:
    """Automates model promotion with validation and approval workflows."""

    def __init__(
        self,
        registry_manager: Optional[ModelRegistryManager] = None,
        tracking_uri: Optional[str] = None,
    ):
        """Initialize model promoter.

        Args:
            registry_manager: ModelRegistryManager instance
            tracking_uri: MLflow tracking URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.registry = registry_manager or ModelRegistryManager(tracking_uri)
        self.client = MlflowClient()

        logger.info("Initialized ModelPromoter")

    def evaluate_candidate(
        self,
        model_name: str,
        candidate_version: str,
        evaluation_metrics: List[str],
        validation_data_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a candidate model version.

        Args:
            model_name: Registered model name
            candidate_version: Version to evaluate
            evaluation_metrics: List of metric names to evaluate
            validation_data_uri: URI to validation data (optional)

        Returns:
            Evaluation results

        Raises:
            Exception: If evaluation fails
        """
        try:
            logger.info(
                "Evaluating candidate model",
                model_name=model_name,
                version=candidate_version,
                metrics=evaluation_metrics,
            )

            # Get candidate model version
            candidate = self.registry.get_model_version(model_name, version=candidate_version)

            # Get metrics from the run
            run = self.client.get_run(candidate.run_id)
            candidate_metrics = {}

            for metric_name in evaluation_metrics:
                metric_value = run.data.metrics.get(metric_name)
                if metric_value is not None:
                    candidate_metrics[metric_name] = metric_value
                else:
                    logger.warning(f"Metric '{metric_name}' not found for candidate version")

            evaluation_result = {
                "model_name": model_name,
                "version": candidate_version,
                "run_id": candidate.run_id,
                "metrics": candidate_metrics,
                "stage": candidate.current_stage,
                "evaluation_timestamp": datetime.utcnow().isoformat(),
            }

            # Add validation data results if provided
            if validation_data_uri:
                logger.info("Running validation on provided dataset")
                # Placeholder for custom validation logic
                evaluation_result["validation_data_uri"] = validation_data_uri

            logger.info(
                "Candidate evaluation completed",
                model_name=model_name,
                version=candidate_version,
                metrics=candidate_metrics,
            )

            return evaluation_result

        except Exception as e:
            logger.error("Failed to evaluate candidate", model_name=model_name, version=candidate_version, error=str(e))
            raise

    def compare_models(
        self,
        model_name: str,
        candidate_version: str,
        baseline_version: Optional[str] = None,
        baseline_stage: Optional[str] = None,
        comparison_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare candidate model with baseline.

        Args:
            model_name: Registered model name
            candidate_version: Candidate version
            baseline_version: Baseline version (or use baseline_stage)
            baseline_stage: Baseline stage (e.g., "Production")
            comparison_metrics: Metrics to compare (uses all if None)

        Returns:
            Comparison results

        Raises:
            ValueError: If neither baseline_version nor baseline_stage provided
        """
        if not baseline_version and not baseline_stage:
            raise ValueError("Must provide either baseline_version or baseline_stage")

        try:
            logger.info(
                "Comparing models",
                model_name=model_name,
                candidate_version=candidate_version,
                baseline_version=baseline_version,
                baseline_stage=baseline_stage,
            )

            # Get candidate metrics
            candidate = self.registry.get_model_version(model_name, version=candidate_version)
            candidate_run = self.client.get_run(candidate.run_id)
            candidate_metrics = candidate_run.data.metrics

            # Get baseline metrics
            if baseline_version:
                baseline = self.registry.get_model_version(model_name, version=baseline_version)
            else:
                baseline = self.registry.get_model_version(model_name, stage=baseline_stage)

            baseline_run = self.client.get_run(baseline.run_id)
            baseline_metrics = baseline_run.data.metrics

            # Determine metrics to compare
            if comparison_metrics:
                metrics_to_compare = comparison_metrics
            else:
                metrics_to_compare = list(set(candidate_metrics.keys()) & set(baseline_metrics.keys()))

            # Compare metrics
            comparison_results = {
                "model_name": model_name,
                "candidate_version": candidate_version,
                "baseline_version": baseline.version,
                "baseline_stage": baseline.current_stage,
                "comparison_timestamp": datetime.utcnow().isoformat(),
                "metric_comparison": {},
                "improvements": {},
            }

            for metric in metrics_to_compare:
                candidate_value = candidate_metrics.get(metric)
                baseline_value = baseline_metrics.get(metric)

                if candidate_value is not None and baseline_value is not None:
                    difference = candidate_value - baseline_value
                    percent_change = (difference / baseline_value * 100) if baseline_value != 0 else 0

                    comparison_results["metric_comparison"][metric] = {
                        "candidate": candidate_value,
                        "baseline": baseline_value,
                        "difference": difference,
                        "percent_change": percent_change,
                    }

                    # Determine if this is an improvement (assuming higher is better)
                    if difference > 0:
                        comparison_results["improvements"][metric] = percent_change

            # Overall recommendation
            num_improvements = len(comparison_results["improvements"])
            num_metrics = len(comparison_results["metric_comparison"])

            comparison_results["recommendation"] = (
                "promote" if num_improvements >= num_metrics * 0.6 else "reject"
            )

            logger.info(
                "Model comparison completed",
                model_name=model_name,
                improvements=num_improvements,
                total_metrics=num_metrics,
                recommendation=comparison_results["recommendation"],
            )

            return comparison_results

        except Exception as e:
            logger.error("Failed to compare models", model_name=model_name, error=str(e))
            raise

    def promote_to_staging(
        self,
        model_name: str,
        version: str,
        require_approval: bool = False,
        auto_evaluate: bool = True,
        evaluation_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Promote model to Staging stage.

        Args:
            model_name: Registered model name
            version: Version to promote
            require_approval: Require manual approval
            auto_evaluate: Automatically evaluate before promotion
            evaluation_metrics: Metrics for evaluation

        Returns:
            Promotion results

        Raises:
            ValueError: If validation fails
        """
        try:
            logger.info(
                "Promoting model to Staging",
                model_name=model_name,
                version=version,
                require_approval=require_approval,
            )

            promotion_result = {
                "model_name": model_name,
                "version": version,
                "target_stage": "Staging",
                "timestamp": datetime.utcnow().isoformat(),
                "approved": not require_approval,
            }

            # Auto-evaluate if requested
            if auto_evaluate and evaluation_metrics:
                evaluation = self.evaluate_candidate(model_name, version, evaluation_metrics)
                promotion_result["evaluation"] = evaluation

                # Check if metrics meet minimum thresholds (placeholder logic)
                if not evaluation.get("metrics"):
                    raise ValueError("Model evaluation failed - no metrics available")

            # Check if approval required
            if require_approval:
                logger.info("Manual approval required for promotion to Staging")
                promotion_result["status"] = "pending_approval"
                promotion_result["approved"] = False

                # Add approval metadata
                self.registry.update_model_metadata(
                    model_name,
                    version,
                    tags={
                        "promotion_status": "pending_staging_approval",
                        "promotion_requested_at": datetime.utcnow().isoformat(),
                    },
                )

                return promotion_result

            # Promote to Staging
            updated_version = self.registry.transition_stage(
                model_name,
                version,
                "Staging",
                description="Automated promotion to Staging",
            )

            promotion_result["status"] = "promoted"
            promotion_result["new_stage"] = updated_version.current_stage

            logger.info("Model promoted to Staging successfully", model_name=model_name, version=version)

            return promotion_result

        except Exception as e:
            logger.error("Failed to promote model to Staging", model_name=model_name, version=version, error=str(e))
            raise

    def promote_to_production(
        self,
        model_name: str,
        version: str,
        require_approval: bool = True,
        compare_with_production: bool = True,
        comparison_metrics: Optional[List[str]] = None,
        min_improvement_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Promote model to Production stage with validation.

        Args:
            model_name: Registered model name
            version: Version to promote
            require_approval: Require manual approval
            compare_with_production: Compare with current production model
            comparison_metrics: Metrics for comparison
            min_improvement_threshold: Minimum improvement percentage required

        Returns:
            Promotion results

        Raises:
            ValueError: If validation fails
        """
        try:
            logger.info(
                "Promoting model to Production",
                model_name=model_name,
                version=version,
                require_approval=require_approval,
                compare_with_production=compare_with_production,
            )

            promotion_result = {
                "model_name": model_name,
                "version": version,
                "target_stage": "Production",
                "timestamp": datetime.utcnow().isoformat(),
                "approved": not require_approval,
            }

            # Compare with current production model if requested
            if compare_with_production:
                try:
                    comparison = self.compare_models(
                        model_name,
                        version,
                        baseline_stage="Production",
                        comparison_metrics=comparison_metrics,
                    )
                    promotion_result["comparison"] = comparison

                    # Check if meets improvement threshold
                    if comparison.get("recommendation") == "reject":
                        raise ValueError(
                            f"Model does not meet improvement criteria. "
                            f"Recommendation: {comparison.get('recommendation')}"
                        )

                    # Check minimum improvement threshold
                    improvements = comparison.get("improvements", {})
                    if improvements:
                        avg_improvement = sum(improvements.values()) / len(improvements)
                        if avg_improvement < min_improvement_threshold:
                            raise ValueError(
                                f"Average improvement ({avg_improvement:.2f}%) below threshold "
                                f"({min_improvement_threshold:.2f}%)"
                            )

                except ValueError as ve:
                    if "No version found" in str(ve):
                        logger.info("No current production model found, skipping comparison")
                    else:
                        raise

            # Check if approval required
            if require_approval:
                logger.info("Manual approval required for promotion to Production")
                promotion_result["status"] = "pending_approval"
                promotion_result["approved"] = False

                # Add approval metadata
                self.registry.update_model_metadata(
                    model_name,
                    version,
                    tags={
                        "promotion_status": "pending_production_approval",
                        "promotion_requested_at": datetime.utcnow().isoformat(),
                    },
                )

                return promotion_result

            # Promote to Production
            updated_version = self.registry.transition_stage(
                model_name,
                version,
                "Production",
                archive_existing=True,
                description="Automated promotion to Production",
            )

            promotion_result["status"] = "promoted"
            promotion_result["new_stage"] = updated_version.current_stage

            logger.info("Model promoted to Production successfully", model_name=model_name, version=version)

            return promotion_result

        except Exception as e:
            logger.error(
                "Failed to promote model to Production",
                model_name=model_name,
                version=version,
                error=str(e),
            )
            raise

    def approve_promotion(self, model_name: str, version: str, approver: str) -> Dict[str, Any]:
        """Approve a pending model promotion.

        Args:
            model_name: Registered model name
            version: Version to approve
            approver: Name/ID of approver

        Returns:
            Approval results
        """
        try:
            logger.info("Approving model promotion", model_name=model_name, version=version, approver=approver)

            # Get model version
            model_version = self.registry.get_model_version(model_name, version=version)

            # Check promotion status
            promotion_status = model_version.tags.get("promotion_status", "")

            if "pending" not in promotion_status:
                raise ValueError(f"No pending promotion for model {model_name} version {version}")

            # Determine target stage
            if "staging" in promotion_status:
                target_stage = "Staging"
            elif "production" in promotion_status:
                target_stage = "Production"
            else:
                raise ValueError(f"Unknown promotion status: {promotion_status}")

            # Update approval metadata
            self.registry.update_model_metadata(
                model_name,
                version,
                tags={
                    "promotion_status": f"approved_for_{target_stage.lower()}",
                    "approved_by": approver,
                    "approved_at": datetime.utcnow().isoformat(),
                },
            )

            # Promote to target stage
            updated_version = self.registry.transition_stage(
                model_name,
                version,
                target_stage,
                archive_existing=True,
                description=f"Approved by {approver}",
            )

            approval_result = {
                "model_name": model_name,
                "version": version,
                "target_stage": target_stage,
                "new_stage": updated_version.current_stage,
                "approver": approver,
                "approved_at": datetime.utcnow().isoformat(),
                "status": "approved_and_promoted",
            }

            logger.info("Model promotion approved and executed", model_name=model_name, version=version, stage=target_stage)

            return approval_result

        except Exception as e:
            logger.error("Failed to approve promotion", model_name=model_name, version=version, error=str(e))
            raise

    def reject_promotion(self, model_name: str, version: str, rejector: str, reason: str) -> Dict[str, Any]:
        """Reject a pending model promotion.

        Args:
            model_name: Registered model name
            version: Version to reject
            rejector: Name/ID of rejector
            reason: Rejection reason

        Returns:
            Rejection results
        """
        try:
            logger.info("Rejecting model promotion", model_name=model_name, version=version, rejector=rejector)

            # Update rejection metadata
            self.registry.update_model_metadata(
                model_name,
                version,
                tags={
                    "promotion_status": "rejected",
                    "rejected_by": rejector,
                    "rejected_at": datetime.utcnow().isoformat(),
                    "rejection_reason": reason,
                },
            )

            rejection_result = {
                "model_name": model_name,
                "version": version,
                "rejector": rejector,
                "rejected_at": datetime.utcnow().isoformat(),
                "reason": reason,
                "status": "rejected",
            }

            logger.info("Model promotion rejected", model_name=model_name, version=version)

            return rejection_result

        except Exception as e:
            logger.error("Failed to reject promotion", model_name=model_name, version=version, error=str(e))
            raise
