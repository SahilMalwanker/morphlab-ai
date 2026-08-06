#Requires -Version 7
# Searches the working tree, every object in the local Git database and the build output for the
# API_KEY and ENDPOINT values from .env. Prints only PASS or FAIL; never prints values or matches.
param(
  [string[]]$Extra = @()
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path -LiteralPath '.env')) { Write-Output 'env: FAIL (.env not found)'; exit 2 }
$values = @{}
foreach ($line in Get-Content -LiteralPath '.env') {
  if ($line -match '^\s*(API_KEY|ENDPOINT)\s*=(.*)$') { $values[$Matches[1]] = $Matches[2].Trim().Trim('"').Trim("'") }
}
foreach ($name in 'API_KEY', 'ENDPOINT') {
  if (-not $values[$name]) { Write-Output "env: FAIL ($name is missing or empty)"; exit 2 }
}

$needles = [System.Collections.Generic.List[string]]::new()
$needles.Add($values['API_KEY'])
$needles.Add($values['ENDPOINT'])
$needles.Add($values['ENDPOINT'].TrimEnd('/'))
try {
  $endpointHost = ([uri]$values['ENDPOINT']).Host
  $generic = 'api.openai.com', 'api.anthropic.com', 'generativelanguage.googleapis.com'
  if ($endpointHost.Length -ge 8 -and $generic -notcontains $endpointHost) { $needles.Add($endpointHost) }
} catch { }

function Test-Text([string]$text) {
  foreach ($n in $needles) { if ($n -and $text.Contains($n, [StringComparison]::Ordinal)) { return $true } }
  return $false
}
function Test-Files([string[]]$paths) {
  $latin1 = [Text.Encoding]::Latin1
  foreach ($p in $paths) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { continue }
    if (Test-Text $latin1.GetString([IO.File]::ReadAllBytes($p))) { return $true }
  }
  return $false
}
function Write-Result([string]$area, [bool]$leaked) { Write-Output ('{0}: {1}' -f $area, ($(if ($leaked) { 'FAIL' } else { 'PASS' }))) }

$failed = $false
Write-Output 'Leak scan (values are read from .env and never printed)'

$tree = @(git ls-files --cached --others --exclude-standard) | Where-Object { $_ -ne '.env' }
$leak = Test-Files $tree
Write-Result 'working tree' $leak; $failed = $failed -or $leak

$objects = (git cat-file --batch-all-objects --batch) -join "`n"
$leak = Test-Text $objects
Write-Result 'git objects (all refs, reflog and unreachable)' $leak; $failed = $failed -or $leak

$buildDirs = @('dist') + $Extra | Where-Object { Test-Path -LiteralPath $_ }
if ($buildDirs.Count -eq 0) {
  Write-Output 'build output: not present'
} else {
  $files = $buildDirs | ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File | ForEach-Object FullName }
  $leak = Test-Files $files
  Write-Result ('build output (' + ($buildDirs -join ', ') + ')') $leak; $failed = $failed -or $leak
}

if ($failed) { exit 1 }
