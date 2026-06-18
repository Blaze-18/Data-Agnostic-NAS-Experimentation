# Move the original dataset into the standardized data/raw folder
$src = "F:\Thesis\Experimentation\Dataset\NAS-Bench-201-v1_0-e61699.pth"
$dstDir = "F:\Thesis\Experimentation\data\raw"
New-Item -ItemType Directory -Force -Path $dstDir | Out-Null
if (Test-Path $src) {
    $dst = Join-Path $dstDir (Split-Path $src -Leaf)
    Write-Host "Moving $src -> $dst"
    Move-Item -Force -Path $src -Destination $dst
    Write-Host "Done."
} else {
    Write-Host "Source file not found: $src"
}
