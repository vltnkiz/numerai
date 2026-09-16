<#
.SYNOPSIS
After a live run nobody is using the PC for: shuts it down, asking on screen first if someone is logged on.

.DESCRIPTION
Started detached by Invoke-ZemirLiveRun.ps1 after every run, and waits for the
run to end first.

- Nobody logged on at the console (the BIOS alarm powered the PC on for the
  run): shuts down.
- Someone logged on, and the task's wake timer woke the PC for this run: asks
  on screen. OK shuts down now, Cancel keeps the PC on, and no answer before
  the timeout shuts down. A prompt that can't be shown keeps the PC on: there's
  no telling whether anyone is at it.
- Someone logged on otherwise (a logon run, a run by hand, someone who logged
  on mid-run): leaves the PC on.

The prompt goes to the console session, since the task itself runs in a
session nobody sees. The shutdown is a full one - `shutdown /s` without
/hybrid, whatever the Fast Startup setting - and isn't forced, so an app with
unsaved work can still hold it up.

.PARAMETER RunPid
The run's process, waited for so the task's result is recorded before the PC goes down.

.PARAMETER Woken
The task's wake timer woke the PC for this run.

.PARAMETER NoShutdown
Decide, prompt and log as usual, but never shut down (for testing and dry runs).
#>
param(
    [Parameter(Mandatory)][string]$Log,
    [int]$RunPid,
    [switch]$Woken,
    [int]$TimeoutSeconds = 120,
    [switch]$NoShutdown
)

function Write-Log([string]$message) {
    "$((Get-Date).ToUniversalTime().ToString('s'))Z $message" | Out-File $Log -Append -Encoding utf8
}

Add-Type -Name Console -Namespace ZemirLiveRun -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint WTSGetActiveConsoleSessionId();
[DllImport("wtsapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
public static extern bool WTSSendMessage(IntPtr server, uint sessionId, string title, int titleBytes,
    string message, int messageBytes, uint style, uint timeoutSeconds, out uint response, bool wait);
[DllImport("wtsapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
static extern bool WTSQuerySessionInformation(IntPtr server, uint sessionId, int infoClass,
    out IntPtr buffer, out uint bytes);
[DllImport("wtsapi32.dll")]
static extern void WTSFreeMemory(IntPtr memory);

// The user logged on at the console: "" if nobody is, null if it can't be told.
public static string User() {
    uint sessionId = WTSGetActiveConsoleSessionId();
    IntPtr buffer;
    uint bytes;
    if (sessionId == 0xFFFFFFFF || !WTSQuerySessionInformation(IntPtr.Zero, sessionId, 5, out buffer, out bytes)) {  // WTSUserName
        return null;
    }
    try { return Marshal.PtrToStringUni(buffer); } finally { WTSFreeMemory(buffer); }
}
'@
$console = [ZemirLiveRun.Console]

if ($RunPid) { Wait-Process -Id $RunPid -Timeout 60 -ErrorAction SilentlyContinue }

$user = $console::User()
if ($null -eq $user) {
    Write-Log "after run: can't tell whether anyone is logged on: leaving the PC on"
    exit 1
} elseif ($user -eq '') {
    Write-Log 'after run: nobody logged on, shutting down'
} elseif (-not $Woken) {
    Write-Log "after run: $user is logged on and the PC wasn't woken for this run, leaving it on"
    exit 0
} else {
    $title = 'zemir_01 live run finished'
    $message = "The PC shuts down in $TimeoutSeconds seconds.`n`nOK: shut down now.`nCancel: keep it on."
    $style = 0x50021  # MB_OKCANCEL | MB_ICONQUESTION | MB_SETFOREGROUND | MB_TOPMOST
    $response = [uint32]0
    $shown = $console::WTSSendMessage([IntPtr]::Zero, $console::WTSGetActiveConsoleSessionId(), $title, $title.Length * 2,
        $message, $message.Length * 2, $style, $TimeoutSeconds, [ref]$response, $true)
    $lastError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()

    if (-not $shown) {
        Write-Log "could not show the shutdown prompt (Win32 error $lastError): leaving the PC on"
        exit 1
    } elseif ($response -eq 2) {
        Write-Log 'shutdown prompt: Cancel, leaving the PC on'
        exit 0
    } elseif ($response -eq 1) {
        Write-Log 'shutdown prompt: OK, shutting down'
    } else {
        Write-Log "shutdown prompt: no answer (response $response), shutting down"
    }
}

if ($NoShutdown) {
    Write-Log 'NoShutdown: not shutting down'
    exit 0
}
shutdown.exe /s /t 0
if ($LASTEXITCODE -ne 0) {
    Write-Log "could not shut down: shutdown.exe exited $LASTEXITCODE"
    exit 1
}
