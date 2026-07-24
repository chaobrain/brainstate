# Claude `ccd` PowerShell Shortcut Design

## Goal

Register `ccd` for the current Windows user so it starts Claude Code with permission prompts disabled while preserving any additional command-line arguments.

## Scope

- Target the current user's all-hosts Windows PowerShell profile at `C:\Users\Administrator\Documents\WindowsPowerShell\profile.ps1`.
- Preserve all existing profile content.
- Do not change Command Prompt, PowerShell 7, other users, or repository runtime behavior.

## Implementation

Append the following function only when an equivalent `ccd` definition is not already present:

```powershell
function ccd {
    & claude --dangerously-skip-permissions @args
}
```

A function is required because a PowerShell alias cannot include fixed arguments. The call operator resolves `claude` normally, and `@args` forwards caller-supplied arguments unchanged.

## Verification

Start a fresh Windows PowerShell process that loads the profile and inspect `Get-Command ccd`. Verification succeeds when the command type is `Function` and its definition contains `claude --dangerously-skip-permissions @args`.

Do not launch an interactive Claude session as part of verification.
