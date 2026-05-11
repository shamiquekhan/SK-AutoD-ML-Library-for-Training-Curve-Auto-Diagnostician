from sk_autod.core.models import DiagnosisReport, Finding, Severity
from sk_autod.core.runner import diagnose
from sk_autod.reports.text import render_text
from sk_autod.reports.json import render_json


def quick_check(train_loss, val_loss) -> str:
	report = diagnose(train_loss, val_loss)
	if not report.findings:
		return "[OK] No issues found"

	finding = report.findings[0]
	return f"[{finding.severity.value}] {finding.detector_name}"


class AutoDCallback:
	def __init__(self, min_epochs: int = 5, print_live: bool = False) -> None:
		self.min_epochs = min_epochs
		self.print_live = print_live
		self.train_losses: list[float] = []
		self.val_losses: list[float] = []

	def on_epoch_end(self, epoch: int, train_loss, val_loss) -> None:
		self.train_losses.append(float(train_loss))
		self.val_losses.append(float(val_loss))

		if epoch + 1 < self.min_epochs:
			return

		if self.print_live:
			report = diagnose(self.train_losses, self.val_losses)
			print(render_text(report))

__all__ = [
	"AutoDCallback",
	"DiagnosisReport",
	"Finding",
	"Severity",
	"diagnose",
	"quick_check",
	"render_json",
]
