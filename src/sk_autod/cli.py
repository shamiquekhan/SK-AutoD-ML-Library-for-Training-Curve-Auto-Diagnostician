from __future__ import annotations

import argparse

from sk_autod import diagnose
from sk_autod.reports.json import render_json
from sk_autod.reports.text import render_text


def _parse_curve(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def check(train_loss: str, val_loss: str) -> None:
    train = _parse_curve(train_loss)
    val = _parse_curve(val_loss)
    report = diagnose(train, val)
    print(render_text(report))


def main() -> None:
    parser = argparse.ArgumentParser(prog="sk-autod")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Diagnose training curves")
    check_parser.add_argument("--train-loss", required=True)
    check_parser.add_argument("--val-loss", required=True)
    check_parser.add_argument("--output", choices=["text", "json"], default="text")

    args = parser.parse_args()
    if args.command == "check":
        report = diagnose(_parse_curve(args.train_loss), _parse_curve(args.val_loss))
        if args.output == "json":
            print(render_json(report))
        else:
            print(render_text(report))
