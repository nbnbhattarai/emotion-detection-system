import mlflow


def log_classification_report(report):
    '''
    Log classification dict report to mlflow
    '''
    for report_type, report_n in report.items():
        for label, label_metrics in report_n.items():
            if isinstance(label_metrics, dict):
                for metric, value in label_metrics.items():
                    mlflow.log_metric(
                        f'{report_type}.{label}.{metric}', value)

            elif isinstance(label_metrics, (float, int, str)):
                mlflow.log_metric(f'{report_type}.{label}', label_metrics)
