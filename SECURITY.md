# Security Policy

## Supported Versions
We provide security fixes for the most recent BrainState release and the active development branch.

| Version | Supported |
| ------- | --------- |
| `main` (development) | Yes |
| Latest release on PyPI | Yes |
| Older releases | No |

## Reporting a Vulnerability
Security issues should be reported privately so that we can work on a fix before public disclosure.

1. Prefer opening a private advisory at https://github.com/chaobrain/brainstate/security/advisories/new.
2. If you cannot use GitHub, email chao.brain@qq.com with a short summary, steps to reproduce, impact, and any CVSS scoring suggestions.
3. Please do not create public GitHub issues, pull requests, or social media posts about an unfixed vulnerability.

We aim to acknowledge new reports within three business days and to provide status updates at least weekly until the issue is resolved.

## During Investigation
After triaging the report we will:
- confirm the vulnerability and assess its impact,
- identify affected versions and mitigation options,
- coordinate on a remediation plan and release schedule,
- credit reporters (if desired) in the release notes once a fix is available.

When a fix is ready we publish patched releases and disclose the issue once a majority of users can reasonably upgrade.

## Third-party Dependencies
BrainState depends on several upstream projects (for example JAX, brainunit, and brainevent). If the vulnerability originates in one of those dependencies, please notify the maintainers of that project directly. Let us know as well so we can track the issue and update our guidance.

## Need Help?
If you have questions about this policy or are unsure whether an issue is security-sensitive, contact the maintainers at chao.brain@qq.com before sharing details publicly.
