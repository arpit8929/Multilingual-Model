$env:FILTER_BRANCH_SQUELCH_WARNING = "1"
$env:GIT_AUTHOR_NAME = ""
$env:GIT_AUTHOR_EMAIL = ""

$commitFilter = @'
if [ "$GIT_AUTHOR_EMAIL" = "155011842+smitpatel3505@users.noreply.github.com" ] || [ "$GIT_AUTHOR_NAME" = "smit patel" ]; then 
    skip_commit "$@"
else 
    git commit-tree "$@"
fi
'@

# Write to temp file
$filterScript = [System.IO.Path]::GetTempFileName()
$commitFilter | Out-File -FilePath $filterScript -Encoding ASCII

try {
    git filter-branch --force --commit-filter "`$(Get-Content $filterScript -Raw)" HEAD
} finally {
    Remove-Item $filterScript -Force
}
