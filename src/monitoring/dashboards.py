"""
Dashboard management for ML monitoring metrics.

Integrates with Databricks SQL Analytics for automated dashboard updates.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class DashboardManager:
    """
    Manage automated dashboards for ML monitoring.
    """
    
    def __init__(
        self,
        workspace_url: Optional[str] = None,
        api_token: Optional[str] = None,
        warehouse_id: Optional[str] = None,
        dashboard_path: Optional[str] = None
    ):
        """
        Initialize dashboard manager.
        
        Args:
            workspace_url: Databricks workspace URL
            api_token: Databricks API token
            warehouse_id: SQL warehouse ID
            dashboard_path: Path to store dashboards
        """
        self.workspace_url = workspace_url
        self.api_token = api_token
        self.warehouse_id = warehouse_id
        self.dashboard_path = dashboard_path or "/ml_monitoring_dashboards"
        
        # Dashboard registry
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        self.metrics_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("Initialized DashboardManager")
    
    def create_dashboard(
        self,
        dashboard_name: str,
        metrics: List[str],
        refresh_interval_minutes: int = 15,
        visualizations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Create a new monitoring dashboard.
        
        Args:
            dashboard_name: Name of the dashboard
            metrics: List of metrics to track
            refresh_interval_minutes: Auto-refresh interval
            visualizations: Custom visualization configurations
        
        Returns:
            Dashboard configuration
        """
        dashboard_id = f"dashboard_{dashboard_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Default visualizations if not provided
        if visualizations is None:
            visualizations = self._generate_default_visualizations(metrics)
        
        dashboard_config = {
            'id': dashboard_id,
            'name': dashboard_name,
            'metrics': metrics,
            'refresh_interval_minutes': refresh_interval_minutes,
            'visualizations': visualizations,
            'created_at': datetime.now(),
            'last_updated': datetime.now(),
            'status': 'active'
        }
        
        self.dashboards[dashboard_id] = dashboard_config
        
        # Initialize metrics cache
        self.metrics_cache[dashboard_id] = []
        
        logger.info(f"Created dashboard: {dashboard_name} (ID: {dashboard_id})")
        
        return dashboard_config
    
    def _generate_default_visualizations(
        self,
        metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate default visualizations for metrics.
        
        Args:
            metrics: List of metric names
        
        Returns:
            List of visualization configurations
        """
        visualizations = []
        
        for metric in metrics:
            # Time series visualization
            visualizations.append({
                'type': 'timeseries',
                'metric': metric,
                'title': f"{metric} Over Time",
                'aggregation': 'mean',
                'window': '1h'
            })
            
            # Current value card
            visualizations.append({
                'type': 'card',
                'metric': metric,
                'title': f"Current {metric}",
                'aggregation': 'last'
            })
        
        return visualizations
    
    def update_metrics(
        self,
        dashboard_id: str,
        metrics_data: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Update dashboard with new metrics.
        
        Args:
            dashboard_id: Dashboard identifier
            metrics_data: Dictionary of metric values
            timestamp: Timestamp for metrics
        
        Returns:
            Success status
        """
        if dashboard_id not in self.dashboards:
            logger.error(f"Dashboard not found: {dashboard_id}")
            return False
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store metrics
        metric_record = {
            'timestamp': timestamp,
            'metrics': metrics_data
        }
        
        if dashboard_id not in self.metrics_cache:
            self.metrics_cache[dashboard_id] = []
        
        self.metrics_cache[dashboard_id].append(metric_record)
        
        # Update dashboard timestamp
        self.dashboards[dashboard_id]['last_updated'] = timestamp
        
        # If using Databricks, send to SQL warehouse
        if self.workspace_url and self.api_token:
            self._send_to_databricks(dashboard_id, metric_record)
        
        logger.info(f"Updated metrics for dashboard: {dashboard_id}")
        
        return True
    
    def _send_to_databricks(
        self,
        dashboard_id: str,
        metric_record: Dict[str, Any]
    ) -> bool:
        """
        Send metrics to Databricks SQL warehouse.
        
        Args:
            dashboard_id: Dashboard identifier
            metric_record: Metric data to send
        
        Returns:
            Success status
        """
        try:
            import requests
            
            # Build SQL insert statement
            dashboard = self.dashboards[dashboard_id]
            
            # In production, this would execute SQL queries against
            # Databricks SQL warehouse to insert metrics
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            # Example: Create table if not exists and insert data
            # This is a simplified example
            logger.info(
                f"Would send metrics to Databricks for dashboard: {dashboard_id}"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to send metrics to Databricks: {e}")
            return False
    
    def generate_report(
        self,
        dashboard_id: str,
        time_window: Optional[timedelta] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate a report from dashboard data.
        
        Args:
            dashboard_id: Dashboard identifier
            time_window: Time window for report (None = all data)
            format: Report format ('json', 'html', 'pdf')
        
        Returns:
            Generated report
        """
        if dashboard_id not in self.dashboards:
            logger.error(f"Dashboard not found: {dashboard_id}")
            return {}
        
        dashboard = self.dashboards[dashboard_id]
        metrics_data = self.metrics_cache.get(dashboard_id, [])
        
        # Filter by time window
        if time_window:
            cutoff = datetime.now() - time_window
            metrics_data = [
                m for m in metrics_data
                if m['timestamp'] >= cutoff
            ]
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(metrics_data, dashboard['metrics'])
        
        report = {
            'dashboard_id': dashboard_id,
            'dashboard_name': dashboard['name'],
            'generated_at': datetime.now().isoformat(),
            'time_window': str(time_window) if time_window else "all_time",
            'data_points': len(metrics_data),
            'metrics_summary': aggregated,
            'visualizations': dashboard['visualizations']
        }
        
        if format == "html":
            report = self._generate_html_report(report)
        elif format == "pdf":
            report = self._generate_pdf_report(report)
        
        logger.info(f"Generated {format} report for dashboard: {dashboard_id}")
        
        return report
    
    def _aggregate_metrics(
        self,
        metrics_data: List[Dict[str, Any]],
        metric_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics data.
        
        Args:
            metrics_data: List of metric records
            metric_names: Names of metrics to aggregate
        
        Returns:
            Aggregated statistics
        """
        import numpy as np
        
        aggregated = {}
        
        for metric in metric_names:
            values = [
                record['metrics'].get(metric)
                for record in metrics_data
                if metric in record['metrics']
            ]
            
            if values:
                aggregated[metric] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'p25': float(np.percentile(values, 25)),
                    'p75': float(np.percentile(values, 75)),
                    'p95': float(np.percentile(values, 95)),
                    'current': float(values[-1])
                }
        
        return aggregated
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """
        Generate HTML report.
        
        Args:
            report: Report data
        
        Returns:
            HTML string
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report['dashboard_name']} - Monitoring Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .metric-card {{ 
                    background: #f9f9f9; 
                    border-left: 4px solid #4CAF50; 
                    padding: 15px; 
                    margin: 10px 0; 
                }}
            </style>
        </head>
        <body>
            <h1>{report['dashboard_name']}</h1>
            <p>Generated: {report['generated_at']}</p>
            <p>Time Window: {report['time_window']}</p>
            <p>Data Points: {report['data_points']}</p>
            
            <h2>Metrics Summary</h2>
        """
        
        for metric, stats in report['metrics_summary'].items():
            html += f"""
            <div class="metric-card">
                <h3>{metric}</h3>
                <table>
                    <tr>
                        <th>Statistic</th>
                        <th>Value</th>
                    </tr>
            """
            for stat, value in stats.items():
                html += f"<tr><td>{stat}</td><td>{value:.4f}</td></tr>"
            html += "</table></div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _generate_pdf_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate PDF report (placeholder).
        
        Args:
            report: Report data
        
        Returns:
            Report with PDF generation note
        """
        # In production, would use library like reportlab or weasyprint
        report['note'] = "PDF generation requires additional libraries"
        logger.info("PDF generation requested but not implemented")
        return report
    
    def list_dashboards(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all dashboards.
        
        Args:
            status: Filter by status ('active', 'archived')
        
        Returns:
            List of dashboard summaries
        """
        dashboards = list(self.dashboards.values())
        
        if status:
            dashboards = [d for d in dashboards if d['status'] == status]
        
        return [
            {
                'id': d['id'],
                'name': d['name'],
                'metrics': d['metrics'],
                'created_at': d['created_at'].isoformat(),
                'last_updated': d['last_updated'].isoformat(),
                'status': d['status'],
                'data_points': len(self.metrics_cache.get(d['id'], []))
            }
            for d in dashboards
        ]
    
    def archive_dashboard(self, dashboard_id: str) -> bool:
        """
        Archive a dashboard.
        
        Args:
            dashboard_id: Dashboard identifier
        
        Returns:
            Success status
        """
        if dashboard_id not in self.dashboards:
            logger.error(f"Dashboard not found: {dashboard_id}")
            return False
        
        self.dashboards[dashboard_id]['status'] = 'archived'
        logger.info(f"Archived dashboard: {dashboard_id}")
        
        return True
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """
        Delete a dashboard.
        
        Args:
            dashboard_id: Dashboard identifier
        
        Returns:
            Success status
        """
        if dashboard_id not in self.dashboards:
            logger.error(f"Dashboard not found: {dashboard_id}")
            return False
        
        del self.dashboards[dashboard_id]
        if dashboard_id in self.metrics_cache:
            del self.metrics_cache[dashboard_id]
        
        logger.info(f"Deleted dashboard: {dashboard_id}")
        
        return True
