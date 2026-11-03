$lines = @()
try {
  $b = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health -TimeoutSec 8
  $lines += 'backend:' + $b.StatusCode
  $lines += $b.Content
} catch {
  $lines += 'backend:FAIL'
}
try {
  $f = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173 -TimeoutSec 8
  $lines += 'frontend:' + $f.StatusCode
} catch {
  $lines += 'frontend:FAIL'
}
$lines += 'node:' + (node --version)
$lines += 'nodemodules:' + (Test-Path 'c:\Users\Sanjay S\forest_fire_ai\frontend\node_modules')
$lines | Out-File -FilePath 'portcheck.txt' -Encoding utf8
