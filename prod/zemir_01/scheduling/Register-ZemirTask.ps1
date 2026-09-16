<#
.SYNOPSIS
Registers (or re-registers) the Windows scheduled task that runs zemir_01's live pipeline (issue #69).

.DESCRIPTION
Run once, interactively, from this checkout - it prompts for your Windows
account password (for a Microsoft account, the account password, not the PIN).
The task runs as you whether or not you're logged on, which is what gives it
Git Credential Manager and `gh` access for the score-log push and failure issues.
Re-running replaces the existing task.

Triggers (any number of invocations is safe - zemir.schedule decides whether
there is a round to run):
  - Daily at 12:00 UTC, waking the machine, and starting as soon as possible
    after a missed start (the machine was off or asleep). No boot trigger:
    the missed-start catch-up covers a late power-on, and with Fast Startup
    on a power-on is a hibernate-resume, which never fires one.
  - At your logon, as a backup catch-up.
#>
param([string]$TaskName = 'zemir_01 live run')

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$wrapper = Join-Path $PSScriptRoot 'Invoke-ZemirLiveRun.ps1'
$user = "$env:USERDOMAIN\$env:USERNAME"

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$wrapper`"" `
    -WorkingDirectory $repo

$daily = New-ScheduledTaskTrigger -Daily -At '12:00'
# A 'Z' start boundary pins the trigger to 12:00 UTC ("synchronize across time
# zones"), so it doesn't drift an hour at each DST change.
$daily.StartBoundary = (Get-Date).ToUniversalTime().Date.AddHours(12).ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -WakeToRun `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 150) `
    -Priority 7

$credential = Get-Credential -UserName $user -Message "Password for $user, so '$TaskName' can run while you're logged off"

Register-ScheduledTask -TaskName $TaskName `
    -Description 'zemir_01 Numerai Classic live run: Invoke-ZemirLiveRun.ps1 (issue #69).' `
    -Action $action -Trigger $daily, $logon -Settings $settings `
    -User $credential.UserName -Password $credential.GetNetworkCredential().Password `
    -RunLevel Limited -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Select-Object TaskName, NextRunTime
