<#
.SYNOPSIS
Scheduled entrypoint for zemir_01's live run (issue #69). Registered by Register-ZemirTask.ps1.

.DESCRIPTION
Everything around the pipeline that isn't the pipeline: checks the checkout is
a clean `main`, pulls, installs, runs scripts/run_pipeline.py, commits the
score log, and opens a GitHub issue when something fails. Whether there is a
round to run at all is decided in Python (zemir.schedule), so this is safe to
start from any trigger, any number of times.

Runs against this checkout directly, not a separate clone: a day this checkout
is on another branch or has uncommitted changes is a failed, retried run, never
a submission of work in progress.

.PARAMETER DryRun
Smoke-run the same path without submitting (run_experiment.py --smoke): checks
the checkout, pull, install and Python environment, not the scores. Never
commits, writes round markers, or opens issues.

.PARAMETER TaskName
The scheduled task this runs as. When its wake timer woke the machine for this
run, Suspend-AfterRun.ps1 asks on screen afterwards and goes back to sleep
unless told not to.
#>
param([switch]$DryRun, [string]$TaskName = 'zemir_01 live run')

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$zemir = Join-Path $repo 'prod\zemir_01'
$python = Join-Path $repo '.venv\Scripts\python.exe'
$scoreLog = 'prod/zemir_01/score_log.jsonl'
$stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'")
$logDir = Join-Path $zemir "runs\scheduled\$stamp"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir 'run.log'
Set-Location $repo
# Every native call below is checked by exit code. Under 'Stop', Windows
# PowerShell 5.1 turns any stderr output from git or gh into a terminating error.
$ErrorActionPreference = 'Continue'

# Exit codes from scripts/run_pipeline.py.
$pipelineFailures = @{
    2 = 'validation gate failed'
    3 = 'round did not open'
    4 = 'not enough available memory'
    5 = 'round changed during the run'
}

function Write-Log([string]$message) {
    "$((Get-Date).ToUniversalTime().ToString('s'))Z $message" | Out-File $log -Append -Encoding utf8
}

# Through cmd.exe so a native tool's stdout and stderr (git progress, tqdm bars)
# land in the log interleaved, as text. Started via .NET rather than `& cmd.exe`
# because PowerShell 5.1 mangles embedded quotes in native arguments; `/s` makes
# cmd strip exactly the outer pair of quotes around the command line.
function Invoke-Logged([string]$commandLine) {
    Write-Log "> $commandLine"
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo 'cmd.exe'
    $startInfo.Arguments = "/d /s /c `"$commandLine >> `"$log`" 2>&1`""
    $startInfo.WorkingDirectory = $repo
    $startInfo.UseShellExecute = $false
    $process = [System.Diagnostics.Process]::Start($startInfo)
    $process.WaitForExit()
    return $process.ExitCode
}

function Resolve-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    return Join-Path $env:USERPROFILE '.local\bin\uv.exe'
}

# One open issue per failure type per round (per UTC day when the round isn't
# known yet); a repeat comments on it instead of opening another.
function Publish-Failure([string]$failureType) {
    Write-Log "FAILED: $failureType"
    if ($DryRun) { return }
    $roundLine = Select-String -Path $log -Pattern '^ZEMIR_ROUND=(\d+)' | Select-Object -Last 1
    $key = if ($roundLine) { "round $($roundLine.Matches[0].Groups[1].Value)" } else { (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd') }
    $title = "Live run failed: $failureType ($key)"
    $tail = (Get-Content $log -Encoding utf8 | Where-Object { $_ -notmatch '\d+%\|' } | Select-Object -Last 60) -join "`n"
    $bodyFile = Join-Path $logDir 'issue_body.md'
    @(
        "Scheduled live run ``$stamp`` on $env:COMPUTERNAME failed: **$failureType**."
        ''
        "Full log: ``$log``"
        ''
        '```text'
        $tail
        '```'
    ) -join "`n" | Out-File $bodyFile -Encoding utf8
    # Native failures don't throw under 'Continue', so each gh call's exit code is checked.
    try {
        $listed = gh issue list --state open --search "`"$title`" in:title" --json number,title
        if ($LASTEXITCODE -ne 0) { throw "gh issue list exited $LASTEXITCODE" }
        $existing = $listed | ConvertFrom-Json | Where-Object { $_.title -eq $title } | Select-Object -First 1
        if ($existing) {
            gh issue comment $existing.number --body-file $bodyFile | Out-Null
        } else {
            gh issue create --title $title --label bug --body-file $bodyFile | Out-Null
        }
        if ($LASTEXITCODE -ne 0) { throw "gh exited $LASTEXITCODE" }
    } catch {
        Write-Log "could not publish the failure to GitHub: $_"
    }
}

# Every exit goes through here. The sleep prompt runs detached, so this task has
# ended before the machine sleeps.
function Exit-Run([int]$code) {
    if ($woken) {
        Write-Log 'woken for this run: starting the sleep prompt'
        Start-Process powershell.exe -WindowStyle Hidden -ArgumentList (
            "-NoProfile -NonInteractive -ExecutionPolicy Bypass " +
            "-File `"$(Join-Path $PSScriptRoot 'Suspend-AfterRun.ps1')`" -Log `"$log`"")
    }
    exit $code
}

# Keep the machine awake while this process lives; Windows drops the request on exit.
$power = Add-Type -Name Power -Namespace ZemirLiveRun -PassThru -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
[void]$power::SetThreadExecutionState([uint32]2147483649)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
# Children inherit it: the fit uses every core, and this box is also a desktop.
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$env:PYTHONUTF8 = '1'

Write-Log "zemir_01 scheduled run$(if ($DryRun) { ' (dry run)' }) in $repo"

# Did this task's wake timer bring the machine out of sleep? The wake event names
# the task ("NT TASK\<name>") whatever the display language.
$woken = Get-WinEvent -MaxEvents 1 -ErrorAction SilentlyContinue -FilterHashtable @{
    LogName = 'System'; ProviderName = 'Microsoft-Windows-Power-Troubleshooter'; Id = 1; StartTime = (Get-Date).AddMinutes(-15)
} | Where-Object { $_.Message -like "*NT TASK\$TaskName*" }

$branch = (git symbolic-ref --short HEAD 2>$null)
$dirty = (git status --porcelain)
if ($branch -ne 'main' -or $dirty) {
    Write-Log "checkout is on '$branch' with $(@($dirty).Count) uncommitted change(s)"
    Publish-Failure 'checkout is not a clean main'
    Exit-Run 1
}
if ((Invoke-Logged 'git pull --ff-only') -ne 0) { Publish-Failure 'git pull failed'; Exit-Run 1 }

$uv = Resolve-Uv
if ((Invoke-Logged "`"$uv`" pip install --python `"$python`" -e prod\zemir_01") -ne 0) {
    Publish-Failure 'dependency install failed'
    Exit-Run 1
}
& $uv pip freeze --python $python 2>$null | Out-File (Join-Path $logDir 'pip_freeze.txt') -Encoding utf8

$script = if ($DryRun) { 'scripts\run_experiment.py --model ensemble --smoke' } else { 'scripts\run_pipeline.py --model ensemble' }
$exitCode = Invoke-Logged "cd /d `"$zemir`" && `"$python`" -u $script"
Write-Log "pipeline exit code $exitCode"
# Round changed mid-fit (e.g. a logon just before noon fitting a stale round):
# the new round is open and unfinished, and the task never starts a second
# instance to catch it, so run again now. live.parquet is re-downloaded.
if (-not $DryRun -and $exitCode -eq 5) {
    $exitCode = Invoke-Logged "cd /d `"$zemir`" && `"$python`" -u $script"
    Write-Log "pipeline exit code $exitCode (rerun for the new round)"
}

if ($DryRun) {
    Exit-Run $exitCode
}

# Whatever the pipeline's outcome - a gate failure still appends its scores (issue #68).
git add -- $scoreLog 2>$null
git diff --cached --quiet -- $scoreLog 2>$null
if ($LASTEXITCODE -ne 0) {
    # Pathspec commit: only the score log, whatever else might be staged.
    $pushed = (Invoke-Logged "git commit -m `"Record score_log entry for scheduled run $stamp`" -- $scoreLog") -eq 0 -and
        ((Invoke-Logged 'git push') -eq 0 -or
         ((Invoke-Logged 'git pull --rebase') -eq 0 -and (Invoke-Logged 'git push') -eq 0))
    if (-not $pushed) { Publish-Failure 'score log push failed' }
}

if ($exitCode -ne 0) {
    $failureType = if ($pipelineFailures.ContainsKey($exitCode)) { $pipelineFailures[$exitCode] } else { "pipeline crashed (exit $exitCode)" }
    Publish-Failure $failureType
}
Exit-Run $exitCode
