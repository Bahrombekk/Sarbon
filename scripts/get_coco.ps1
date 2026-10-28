# Download and extract COCO 2017 (train + val images, annotations). Resumable.
# Usage: powershell -File scripts\get_coco.ps1 [-Root D:\data\coco]
param([string]$Root = "D:\data\coco")
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Root | Out-Null
$files = @(
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/zips/train2017.zip"
)
foreach ($u in $files) {
    $zip = Join-Path $Root (Split-Path $u -Leaf)
    $done = "$zip.extracted"
    if (Test-Path $done) { Write-Output "skip $u"; continue }
    Write-Output "download $u"
    & curl.exe -L -C - --retry 20 --retry-delay 5 --retry-all-errors -s -S -o $zip $u
    if ($LASTEXITCODE -ne 0) { throw "download failed: $u" }
    Write-Output "extract $zip"
    & tar.exe -xf $zip -C $Root
    if ($LASTEXITCODE -ne 0) { throw "extract failed: $zip" }
    New-Item -ItemType File $done | Out-Null
    Remove-Item $zip
}
Write-Output "COCO ready in $Root"
