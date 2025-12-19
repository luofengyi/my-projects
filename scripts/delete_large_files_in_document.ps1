<#
.SYNOPSIS
  扫描指定 document 目录，删除大于阈值（默认 100MB）的文件。

.DESCRIPTION
  递归遍历目录下所有文件，找出超过阈值的文件并删除。
  默认会执行删除；如需仅预览不删除，请使用 -DryRun。

.EXAMPLE
  # 预览（不删除）
  powershell -ExecutionPolicy Bypass -File .\scripts\delete_large_files_in_document.ps1 -DryRun

.EXAMPLE
  # 删除（默认）
  powershell -ExecutionPolicy Bypass -File .\scripts\delete_large_files_in_document.ps1

.EXAMPLE
  # 自定义路径与阈值（200MB）
  powershell -ExecutionPolicy Bypass -File .\scripts\delete_large_files_in_document.ps1 -DocumentPath ".\MERC-main\JOYFUL\document" -ThresholdMB 200
#>

param(
  [string]$DocumentPath = ".\MERC-main\JOYFUL\document",
  [int]$ThresholdMB = 100,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Format-Bytes([long]$bytes) {
  if ($bytes -ge 1GB) { return ("{0:N2} GB" -f ($bytes / 1GB)) }
  if ($bytes -ge 1MB) { return ("{0:N2} MB" -f ($bytes / 1MB)) }
  if ($bytes -ge 1KB) { return ("{0:N2} KB" -f ($bytes / 1KB)) }
  return ("{0} B" -f $bytes)
}

$thresholdBytes = [long]$ThresholdMB * 1MB
$fullPath = (Resolve-Path -LiteralPath $DocumentPath).Path

Write-Host ("[Info] Scan path: {0}" -f $fullPath)
Write-Host ("[Info] Threshold: > {0} ({1} bytes)" -f ("$ThresholdMB MB"), $thresholdBytes)
Write-Host ("[Info] Mode: {0}" -f ($(if ($DryRun) { "DryRun (no delete)" } else { "Delete" })))

$largeFiles = Get-ChildItem -LiteralPath $fullPath -Recurse -File -Force |
  Where-Object { $_.Length -gt $thresholdBytes } |
  Sort-Object Length -Descending

if (-not $largeFiles -or $largeFiles.Count -eq 0) {
  Write-Host "[OK] No files larger than threshold."
  exit 0
}

Write-Host ("[Found] {0} file(s) larger than threshold:" -f $largeFiles.Count)
foreach ($f in $largeFiles) {
  Write-Host ("- {0}  ({1})" -f $f.FullName, (Format-Bytes $f.Length))
}

if ($DryRun) {
  Write-Host "[DryRun] Finished. No files were deleted."
  exit 0
}

Write-Host "[Delete] Deleting..."
foreach ($f in $largeFiles) {
  try {
    Remove-Item -LiteralPath $f.FullName -Force
    Write-Host ("  Deleted: {0}" -f $f.FullName)
  } catch {
    Write-Host ("  Failed:  {0}" -f $f.FullName)
    Write-Host ("          {0}" -f $_.Exception.Message)
  }
}

Write-Host "[Done] Large file deletion completed."





