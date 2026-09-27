# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities via private security advisories or by opening a draft security advisory on GitHub.

Please include:
- Description and nature of the issue
- Affected components (e.g. `src/classone/server/app.py`, `src/classone/modeling/`)
- Proof-of-concept or steps to reproduce
- Potential impact assessment

## Security Practices

This project adheres to OpenSSF and SLSA supply chain best practices:
- Dependency audits are executed automatically on every commit via `uv audit`.
- GitHub Actions workflows use pinned commit SHAs and strict read-only token permissions.
- Docker containers run as non-root users (`UID 1001`).
