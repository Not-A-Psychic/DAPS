Set-Location "S:\UMIch\f24\IS\proj\mkiii\data\PKLot.v1-raw.voc\test\annots"

$files = Get-ChildItem -File | Sort-Object Name

$counter = 0

foreach ($file in $files) {
    $newName = "{0:D5}.xml" -f $counter

    Rename-Item -Path $file.FullName -NewName $newName

    $counter++
}

Write-Output "Renaming completed!"
