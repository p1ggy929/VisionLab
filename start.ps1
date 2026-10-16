#Requires -Version 5.1
<#
    视界智研 —— 基于双目视觉测距与目标检测算法实验平台 · 一键启动

    用法：
        .\start.ps1                      使用默认设置启动
        .\start.ps1 -Port 8080           指定端口
        .\start.ps1 -Backend mock        强制使用合成场景（无相机演示）
        .\start.ps1 -Backend replay      使用离线回放（需已有回放样本）
        .\start.ps1 -NoCamera            等价于 -Backend mock
        .\start.ps1 -NoBrowser           不自动打开浏览器
        .\start.ps1 -Debug               开启调试模式
        .\start.ps1 -Check               只做环境自检，不启动服务
        .\start.ps1 -Python "C:\Python312\python.exe"   指定解释器

    如果提示「禁止运行脚本」（执行策略限制），请改用：
        powershell -ExecutionPolicy Bypass -File .\start.ps1
    或直接双击 start.bat（批处理不受执行策略限制）。

    重要：本文件必须以「UTF-8 带 BOM」保存。Windows PowerShell 5.1 在无 BOM 时
    会按系统 ANSI 代码页解码 .ps1，中文会被解成乱码字节并破坏语法结构。
#>
# 说明：这里刻意不使用 [CmdletBinding()]，因为它会自动注入通用参数 -Debug，
# 与本脚本自定义的 -Debug 冲突（报 “A parameter with the name 'Debug' was
# defined multiple times for the command”）。
param(
    [int]$Port = 0,
    [string]$BindHost = "",
    [string]$Backend = "",
    [switch]$NoCamera,
    [switch]$NoBrowser,
    [switch]$Debug,
    [switch]$Check,
    [string]$Python = ""
)

$ErrorActionPreference = "Continue"

# ---------- 控制台切到 UTF-8，保证 Python 的中文输出不乱码 ----------
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
try { $null = chcp 65001 } catch { }
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# ---------- 定位仓库根目录 ----------
$Root = $PSScriptRoot
if (-not $Root) { $Root = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location -LiteralPath $Root

function Write-Line {
    param([string]$Text = "", [string]$Color = "")
    if ($Color) { Write-Host $Text -ForegroundColor $Color } else { Write-Host $Text }
}

function Show-Banner {
    Write-Line ""
    Write-Line "============================================================" "Cyan"
    Write-Line "  视界智研 —— 基于双目视觉测距与目标检测算法实验平台" "Cyan"
    Write-Line "  VisionLab-D455  v1.0.0" "Cyan"
    Write-Line "============================================================" "Cyan"
}

# ---------- 查找 Python 解释器 ----------
function Test-PythonDeps {
    <# 探测解释器是否具备平台的核心依赖。

       必须用**退出码**判定，不能用“输出里是否含某个标记字符串”：
       原生命令失败时 PowerShell 会把命令行本身回显到错误流，而命令行里
       恰好含有那个标记（例如 print('VLPROBE-OK')），-match 就会误判为通过。
    #>
    param([string]$Path)
    if (-not $Path) { return $false }
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $global:LASTEXITCODE = 0
    & $Path -c "import flask, numpy, cv2, yaml, sqlite3, PIL" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
}

function Get-PythonCandidates {
    <# 按优先级返回候选解释器：显式指定 > 平台自身 venv > 同级原项目 venv > PATH。
       同级目录用通配符扫描，因此脚本中不写死任何非 ASCII 路径。 #>
    param([string]$Explicit)

    $list = New-Object System.Collections.Generic.List[string]
    if ($Explicit) { [void]$list.Add($Explicit) }
    if ($env:VISIONLAB_PYTHON) { [void]$list.Add($env:VISIONLAB_PYTHON) }
    [void]$list.Add((Join-Path $Root ".venv\Scripts\python.exe"))
    [void]$list.Add((Join-Path $Root "venv\Scripts\python.exe"))

    $parent = Split-Path -Parent $Root
    if ($parent -and (Test-Path -LiteralPath $parent)) {
        foreach ($suffix in @(".venv\Scripts\python.exe", "venv\Scripts\python.exe")) {
            $folders = Get-ChildItem -LiteralPath $parent -Directory -ErrorAction SilentlyContinue
            foreach ($folder in $folders) {
                [void]$list.Add((Join-Path $folder.FullName $suffix))
            }
        }
    }
    foreach ($name in @("python", "python3", "py")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { [void]$list.Add($cmd.Source) }
    }
    return $list
}

function Find-PythonInterpreter {
    <# 优先返回能通过依赖探测的候选；都不通过时退回第一个存在的解释器。
       注意不能只看目录是否存在：同级目录下可能还有原项目 B / U-Net_v2 的
       venv，它们装的是另一套依赖（缺少 Flask），选中就会启动失败。 #>
    param([string]$Explicit)

    $candidates = Get-PythonCandidates -Explicit $Explicit
    $firstExisting = $null
    foreach ($item in $candidates) {
        if (-not $item) { continue }
        if (-not (Test-Path -LiteralPath $item)) { continue }
        if (-not $firstExisting) { $firstExisting = $item }
        if (Test-PythonDeps -Path $item) {
            return [pscustomobject]@{ Path = $item; DepsOk = $true }
        }
    }
    if ($firstExisting) {
        return [pscustomobject]@{ Path = $firstExisting; DepsOk = $false }
    }
    return $null
}

Show-Banner

$found = Find-PythonInterpreter -Explicit $Python
if (-not $found) {
    Write-Line ""
    Write-Line "[错误] 找不到可用的 Python 解释器。" "Red"
    Write-Line "       安装 Python 3.10+ 后重试，或用参数指定：" "Yellow"
    Write-Line "         .\start.ps1 -Python `"C:\Python312\python.exe`"" "Yellow"
    exit 1
}
$python = $found.Path

Write-Line ("  解释器   : " + $python)
Write-Line ("  工作目录 : " + $Root)
try {
    $version = (& $python --version 2>&1 | Out-String).Trim()
    Write-Line ("  版本     : " + $version)
} catch { }
if (-not $found.DepsOk) {
    Write-Line "  依赖状态 : 缺少核心依赖（下面会给出提示）" "Yellow"
}

# ---------- 依赖快检 ----------
# 同样用退出码判定；也不能把 $ErrorActionPreference 设成 Stop，否则原生命令
# 写到 stderr 的普通告警会被 PowerShell 当成终止性错误，脚本在第一行就中断。
$depsOk = Test-PythonDeps -Path $python
if (-not $depsOk) {
    Write-Line ""
    Write-Line "[警告] 核心依赖检查未通过（缺少 flask / numpy / cv2 / yaml / pillow 之一）。" "Yellow"
    Write-Line "       安装方式：`"$python`" -m pip install -r requirements.txt" "Yellow"
    Write-Line "       或改用参数指定已配好环境的解释器：.\start.ps1 -Python <path>" "Yellow"
    Write-Line "       仍会尝试启动；若失败请运行：`"$python`" tools\self_check.py" "Yellow"
    Write-Line ""
}

if ($Check) {
    Write-Line ""
    Write-Line "仅执行环境自检（-Check）……" "Cyan"
    & $python (Join-Path $Root "tools\self_check.py")
    exit $LASTEXITCODE
}

# ---------- 组织启动参数 ----------
$pyArgs = New-Object System.Collections.Generic.List[string]
[void]$pyArgs.Add("app.py")
if ($Port -gt 0) { [void]$pyArgs.Add("--port"); [void]$pyArgs.Add("$Port") }
if ($BindHost) { [void]$pyArgs.Add("--host"); [void]$pyArgs.Add($BindHost) }
if ($NoBrowser) { [void]$pyArgs.Add("--no-browser") }
if ($Debug) { [void]$pyArgs.Add("--debug") }

$effectiveBackend = $Backend
if ($NoCamera) { $effectiveBackend = "mock" }
if ($effectiveBackend) {
    $env:VISIONLAB_CAMERA__BACKEND = $effectiveBackend
    Write-Line ("  数据源   : " + $effectiveBackend)
}

$port = if ($Port -gt 0) { $Port } else { 5000 }
Write-Line ("  访问地址 : http://127.0.0.1:" + $port + "/")
Write-Line ("  默认账号 : admin / admin123")
Write-Line ""
Write-Line "  正在启动，按 Ctrl+C 停止……" "DarkGray"
Write-Line ""

# ---------- 启动 ----------
$logFile  = Join-Path $Root "storage\logs\visionlab.log"
$launchAt = Get-Date
& $python @pyArgs
$exitCode = $LASTEXITCODE

# 判定“服务是否真的起来过”：日志里出现不早于本次启动时刻的“平台启动”横幅才算成功。
# 必要性：Ctrl+C 主动停止服务时 Python 同样返回非 0 码（Windows 控制台中断常为
# -1073741510），若一律按启动失败处理，用户每次正常关服务都会看到“[错误] … 退出码”
# 加一整段自检输出，误以为平台根本没启动。只看退出码无法区分这两种情况。
function Test-ServerStarted {
    param([string]$LogPath, [datetime]$Since)
    if (-not (Test-Path -LiteralPath $LogPath)) { return $false }
    try {
        $tail = Get-Content -LiteralPath $LogPath -Tail 400 -Encoding UTF8 -ErrorAction Stop
    } catch { return $false }
    foreach ($ln in $tail) {
        if ($ln -notlike "*平台启动*") { continue }
        $m = [regex]::Match($ln, '^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]')
        if (-not $m.Success) { continue }
        try { $t = [datetime]::Parse($m.Groups[1].Value) } catch { continue }
        if ($t -ge $Since) { return $true }
    }
    return $false
}

if ($exitCode -eq 0) {
    Write-Line ""
    Write-Line "  平台已停止。" "Green"
} elseif (Test-ServerStarted -LogPath $logFile -Since $launchAt) {
    # 服务曾成功启动，非 0 码来自 Ctrl+C 或外部结束，属正常关闭，不再打印自检。
    Write-Line ""
    Write-Line ("  平台已停止（退出码 " + $exitCode + "；Ctrl+C 主动停止属正常）。") "Green"
} else {
    Write-Line ""
    Write-Line ("[错误] 平台未能启动，退出码 " + $exitCode) "Red"
    Write-Line "  下面运行环境自检，帮助定位原因：" "Yellow"
    Write-Line ""
    & $python (Join-Path $Root "tools\self_check.py")
    Write-Line ""
    Write-Line "  常见原因：端口被占用（换 -Port）、依赖缺失、存储目录不可写。" "Yellow"
    Write-Line "  也可直接运行 `"$python`" app.py 查看完整错误堆栈。" "Yellow"
}

exit $exitCode
