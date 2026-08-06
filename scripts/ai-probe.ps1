#Requires -Version 7
# Probes the AI provider configured in .env for tool calling, image input and
# streaming with three minimal requests. Prints only HTTP status codes and the
# shape of each answer; never the key, the endpoint, its host or a response body.
# Usage: pwsh -NoProfile -File scripts/ai-probe.ps1
$ErrorActionPreference = 'Stop'

function Out-Line([string]$text) { Write-Output $text }

try {
  $envFile = Join-Path (Split-Path -Parent $PSScriptRoot) '.env'
  if (-not (Test-Path -LiteralPath $envFile)) { Out-Line 'probe: .env not found'; exit 2 }
  $vars = @{}
  foreach ($line in Get-Content -LiteralPath $envFile) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
      $v = $Matches[2]
      if ($v.Length -ge 2 -and (($v[0] -eq '"' -and $v[-1] -eq '"') -or ($v[0] -eq "'" -and $v[-1] -eq "'"))) { $v = $v.Substring(1, $v.Length - 2) }
      $vars[$Matches[1]] = $v
    }
  }
  $key = $vars['API_KEY']
  $endpoint = $vars['ENDPOINT']
  if (-not $key -or -not $endpoint) { Out-Line 'probe: API_KEY or ENDPOINT is missing in .env'; exit 2 }
  $uri = [Uri]$endpoint
  $shapeOk = $uri.Scheme -eq 'https' -and $uri.Host -match '\.openai\.azure\.com$' -and $uri.AbsolutePath -match '^/openai/deployments/[^/]+/chat/completions/?$' -and $uri.Query -match 'api-version='
  if (-not $shapeOk) { Out-Line 'probe: ENDPOINT is not an Azure OpenAI chat completions URL'; exit 2 }
  $apiVersion = [regex]::Match($uri.Query, 'api-version=([^&]+)').Groups[1].Value
  $limitField = if ([string]::CompareOrdinal($apiVersion, '2024-09-01') -ge 0) { 'max_completion_tokens' } else { 'max_tokens' }
  Out-Line "probe: Azure OpenAI chat completions; api-version from 2024-09-01: $($limitField -eq 'max_completion_tokens')"

  function Send-Probe([hashtable]$body) {
    $json = ConvertTo-Json -InputObject $body -Depth 20 -Compress
    try {
      return Invoke-WebRequest -Uri $uri -Method Post -Headers @{ 'api-key' = $key } -ContentType 'application/json' -Body $json -SkipHttpErrorCheck -TimeoutSec 90
    } catch {
      return [pscustomobject]@{ StatusCode = 0; Content = ''; Headers = @{}; Failure = $_.Exception.GetType().Name }
    }
  }

  function Describe-Failure($r) {
    if ($r.StatusCode -eq 0) { return "no HTTP answer ($($r.Failure))" }
    $code = $null
    try { $code = ($r.Content | ConvertFrom-Json).error.code } catch { }
    if ($code -and $code -match '^[A-Za-z0-9_.-]{1,64}$') { return "HTTP $($r.StatusCode), code $code" }
    return "HTTP $($r.StatusCode)"
  }

  # 1. Tool calling
  $tool = @{ type = 'function'; function = @{ name = 'getRobotState'; description = 'Summarize the measured state of the simulated robot.'; parameters = @{ type = 'object'; properties = @{}; additionalProperties = $false } } }
  $b1 = @{ messages = @(@{ role = 'system'; content = 'You operate a simulated robot only through the tools provided.' }, @{ role = 'user'; content = 'What is the robot doing right now?' }); tools = @($tool); tool_choice = 'auto' }
  $b1[$limitField] = 400
  $r1 = Send-Probe $b1
  if ($r1.StatusCode -eq 200) {
    $j = $r1.Content | ConvertFrom-Json
    $c = $j.choices[0]
    $names = @($c.message.tool_calls | ForEach-Object { $_.function.name }) -join ', '
    Out-Line "tool calling: HTTP 200, finish_reason $($c.finish_reason), tool calls: $(if ($names) { $names } else { 'none' }), model $($j.model), tokens $($j.usage.prompt_tokens)/$($j.usage.completion_tokens)"
  } else { Out-Line "tool calling: $(Describe-Failure $r1)" }

  # 2. Image input: a 32x32 red PNG made here
  Add-Type -AssemblyName System.Drawing
  $bmp = [System.Drawing.Bitmap]::new(32, 32)
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.Clear([System.Drawing.Color]::FromArgb(255, 220, 30, 30))
  $g.Dispose()
  $ms = [System.IO.MemoryStream]::new()
  $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
  $bmp.Dispose()
  $b64 = [Convert]::ToBase64String($ms.ToArray())
  $ms.Dispose()
  $b2 = @{ messages = @(@{ role = 'user'; content = @(@{ type = 'text'; text = 'Which single color fills this image? Answer with one word.' }, @{ type = 'image_url'; image_url = @{ url = "data:image/png;base64,$b64"; detail = 'low' } }) }) }
  $b2[$limitField] = 400
  $r2 = Send-Probe $b2
  if ($r2.StatusCode -eq 200) {
    $j = $r2.Content | ConvertFrom-Json
    $c = $j.choices[0]
    $answer = [string]$c.message.content
    $short = ($answer -replace '[^A-Za-z ]', '').Trim()
    if ($short.Length -gt 20) { $short = $short.Substring(0, 20) }
    Out-Line "image input: HTTP 200, finish_reason $($c.finish_reason), answer `"$short`""
  } else { Out-Line "image input: $(Describe-Failure $r2)" }

  # 3. Streaming (server-sent events)
  $b3 = @{ messages = @(@{ role = 'user'; content = 'Reply with the single word OK.' }); stream = $true }
  $b3[$limitField] = 200
  $r3 = Send-Probe $b3
  if ($r3.StatusCode -eq 200) {
    $type = [string]($r3.Headers['Content-Type'] | Select-Object -First 1)
    $events = ([regex]::Matches([string]$r3.Content, '(?m)^data: ')).Count
    $done = ([string]$r3.Content).Contains('data: [DONE]')
    Out-Line "streaming: HTTP 200, content type $($type -replace ';.*$', ''), $events data events, end marker seen: $done"
  } else { Out-Line "streaming: $(Describe-Failure $r3)" }
} catch {
  Out-Line "probe: stopped ($($_.Exception.GetType().Name))"
  exit 1
}
