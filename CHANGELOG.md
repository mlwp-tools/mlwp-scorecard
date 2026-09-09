# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Initial development of `mlwp-scorecards`, a package for rendering weather
forecasting scorecards from pre-computed verification statistics. A scorecard
compares two prediction sources, each scored against a common truth source, and
colours the difference between their scores across nested groupings of variable,
level, region, metric and forecast lead time.

Design and rationale are documented in [`PLAN.md`](PLAN.md).
