<#
.SYNOPSIS
After a live run the machine was woken for: asks on screen, then goes back to sleep unless told not to.

.DESCRIPTION
Started detached by Invoke-ZemirLiveRun.ps1, so the task has already ended
when the machine sleeps. A task instance still running at the next wake would
make that day's trigger be dropped (MultipleInstances IgnoreNew).

The prompt goes to the console session, since the task itself runs in a
session nobody sees. OK sleeps now, Cancel keeps the machine awake, and no
answer before the timeout sleeps. A prompt that can't be shown keeps the
machine awake: there's no telling whether anyone is at it.

.PARAMETER NoSleep
Show the prompt and log the answer, but never sleep (for testing).
#>
param(
    [Parameter(Mandatory)][string]$Log,
    [int]$TimeoutSeconds = 120,
    [switch]$NoSleep
)

function Write-Log([string]$message) {
    "$((Get-Date).ToUniversalTime().ToString('s'))Z $message" | Out-File $Log -Append -Encoding utf8
}

Add-Type -Name Suspend -Namespace ZemirLiveRun -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint WTSGetActiveConsoleSessionId();
[DllImport("wtsapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
public static extern bool WTSSendMessage(IntPtr server, uint sessionId, string title, int titleBytes,
    string message, int messageBytes, uint style, uint timeoutSeconds, out uint response, bool wait);
[DllImport("powrprof.dll", SetLastError = true)]
public static extern bool SetSuspendState(bool hibernate, bool forceCritical, bool disableWakeEvent);
[DllImport("advapi32.dll", SetLastError = true)]
static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);
[DllImport("advapi32.dll", SetLastError = true)]
static extern bool LookupPrivilegeValue(string system, string name, out long luid);
[DllImport("advapi32.dll", SetLastError = true)]
static extern bool AdjustTokenPrivileges(IntPtr token, bool disableAll, ref TokenPrivilege state,
    int length, IntPtr previous, IntPtr returnLength);
[DllImport("kernel32.dll")]
static extern IntPtr GetCurrentProcess();

[StructLayout(LayoutKind.Sequential, Pack = 1)]
struct TokenPrivilege { public int Count; public long Luid; public int Attributes; }

// SetSuspendState needs SeShutdownPrivilege enabled, not just held.
public static bool EnableShutdownPrivilege() {
    IntPtr token;
    TokenPrivilege state = new TokenPrivilege { Count = 1, Attributes = 2 };  // SE_PRIVILEGE_ENABLED
    return OpenProcessToken(GetCurrentProcess(), 0x28, out token)  // TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY
        && LookupPrivilegeValue(null, "SeShutdownPrivilege", out state.Luid)
        && AdjustTokenPrivileges(token, false, ref state, 0, IntPtr.Zero, IntPtr.Zero)
        && Marshal.GetLastWin32Error() == 0;
}
'@
# Not -PassThru: that also returns the nested TokenPrivilege type.
$power = [ZemirLiveRun.Suspend]

$title ='zemir_01 live run finished'
$message = "The PC goes back to sleep in $TimeoutSeconds seconds.`n`nOK: sleep now.`nCancel: keep it awake."
$style = 0x50021  # MB_OKCANCEL | MB_ICONQUESTION | MB_SETFOREGROUND | MB_TOPMOST
$response = [uint32]0
$shown = $power::WTSSendMessage([IntPtr]::Zero, $power::WTSGetActiveConsoleSessionId(), $title, $title.Length * 2,
    $message, $message.Length * 2, $style, $TimeoutSeconds, [ref]$response, $true)
$lastError = [Runtime.InteropServices.Marshal]::GetLastWin32Error()

if (-not $shown) {
    # Can't tell whether anyone is there, so don't sleep under them.
    Write-Log "could not show the sleep prompt (Win32 error $lastError): staying awake"
    exit 1
} elseif ($response -eq 2) {
    Write-Log 'sleep prompt: Cancel, staying awake'
    exit 0
} elseif ($response -eq 1) {
    Write-Log 'sleep prompt: OK, going back to sleep'
} else {
    Write-Log "sleep prompt: no answer (response $response), going back to sleep"
}

if ($NoSleep) {
    Write-Log 'NoSleep: not sleeping'
    exit 0
}
if (-not $power::EnableShutdownPrivilege()) {
    Write-Log "could not go back to sleep: SeShutdownPrivilege not enabled (Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error()))"
    exit 1
}
# Sleep, not hibernate, with wake timers left armed for the next run.
if (-not $power::SetSuspendState($false, $false, $false)) {
    Write-Log "could not go back to sleep: SetSuspendState failed (Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error()))"
    exit 1
}
