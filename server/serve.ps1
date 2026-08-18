#Requires -Version 7.3
# Serves the built app (dist/) and the AI proxy routes on one local origin with PowerShell 7 only, for
# machines where Node.js may not run. Mirrors server/app.ts and server/azure.ts: same routes, guards,
# limits and sanitized errors. Never prints the key, the endpoint or request and response bodies.
# Usage: pwsh -NoProfile -File server/serve.ps1 [-Port 4173] [-Root dist] [-EnvFile .env]
[CmdletBinding()]
param(
  [ValidateRange(1024, 65535)][int]$Port = 4173,
  [string]$Root,
  [string]$EnvFile,
  # Defines the functions and returns without serving (used by server/serve.tests.ps1).
  [switch]$LibraryOnly
)
Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Root) { $Root = Join-Path $repoRoot 'dist' }
if (-not $EnvFile) { $EnvFile = Join-Path $repoRoot '.env' }

# Keep in step with src/ai/tools.ts and src/ai/protocol.ts (checked by tests/server/servePs1.test.ts).
$ToolNames = @('getRobotCapabilities', 'getRobotState', 'getSceneObjects', 'pointAt', 'navigateTo', 'navigateToPlace', 'rememberPlace', 'forgetPlace', 'executeSkill', 'draftSkill', 'mirrorSkill', 'evaluateSkillDraft', 'improveSkillDraft', 'trainGaitPolicy', 'trainLocomotion', 'locomote', 'runSkillSequence', 'selectRobot', 'pushRobot', 'resetSimulation', 'moveTipTo', 'setJointAngles', 'buildRobot', 'modifyTask', 'cancelTask', 'listSkills', 'saveSkillFromDemonstration', 'evaluateSkill', 'runControllerExperiment', 'saveSkill')
$MaxMessageChars = 16000
$MaxTotalChars = 48000
$MaxMessages = 40
$MaxTools = 24
$MaxToolCalls = 8
$MaxArgumentChars = 4000
$MaxDescriptionChars = 1024
$MinOutputTokens = 16
$MaxOutputTokens = 2048
$DefaultOutputTokens = 1024
$BodyLimitBytes = 256 * 1024
$PerSessionLimit = 20
$OverallLimit = 100
$WindowMs = 60000
$UpstreamTimeoutSeconds = 30
$IdentPattern = '^[A-Za-z0-9_.:-]{1,64}\z'
$ToolNamePattern = '^[A-Za-z0-9_-]{1,64}\z'
$SessionPattern = '^[A-Za-z0-9-]{8,64}\z'
$SafeTokenPattern = '^[A-Za-z0-9_.-]{1,64}\z'

$Csp = "default-src 'self'; script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval'; worker-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
$MimeTypes = @{
  '.html' = 'text/html; charset=utf-8'; '.js' = 'text/javascript; charset=utf-8'; '.mjs' = 'text/javascript; charset=utf-8'
  '.css' = 'text/css; charset=utf-8'; '.wasm' = 'application/wasm'; '.json' = 'application/json'; '.map' = 'application/json'
  '.svg' = 'image/svg+xml'; '.png' = 'image/png'; '.jpg' = 'image/jpeg'; '.ico' = 'image/x-icon'; '.txt' = 'text/plain; charset=utf-8'
}

function New-JsonValue($value) { return [System.Text.Json.Nodes.JsonValue]::Create($value) }

function Get-JsonString($node) {
  if ($node -is [System.Text.Json.Nodes.JsonValue] -and $node.GetValueKind() -eq [System.Text.Json.JsonValueKind]::String) { return $node.GetValue[string]() }
  return $null
}

function Get-JsonNumber($node) {
  if ($node -is [System.Text.Json.Nodes.JsonValue] -and $node.GetValueKind() -eq [System.Text.Json.JsonValueKind]::Number) { return $node.GetValue[double]() }
  return $null
}

function Test-OnlyKeys([System.Text.Json.Nodes.JsonObject]$obj, [string[]]$allowed) {
  foreach ($p in $obj) { if ($p.Key -cnotin $allowed) { return $p.Key } }
  return $null
}

# Reads API_KEY and ENDPOINT; problems name the variable, never its value.
function Read-AiConfig([string]$path) {
  if (-not (Test-Path -LiteralPath $path)) { return @{ Ok = $false; Reason = 'The .env file was not found. Copy .env.example to .env and set API_KEY and ENDPOINT.' } }
  $vars = @{}
  foreach ($line in [System.IO.File]::ReadAllLines($path)) {
    if ($line -cmatch '^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
      $v = $Matches[2]
      if ($v.Length -ge 2 -and (($v[0] -eq '"' -and $v[-1] -eq '"') -or ($v[0] -eq "'" -and $v[-1] -eq "'"))) { $v = $v.Substring(1, $v.Length - 2) }
      $vars[$Matches[1]] = $v.Trim()
    }
  }
  $key = [string]$vars['API_KEY']
  $endpoint = [string]$vars['ENDPOINT']
  if (-not $key) { return @{ Ok = $false; Reason = 'API_KEY is missing or empty in .env.' } }
  if (-not $endpoint) { return @{ Ok = $false; Reason = 'ENDPOINT is missing or empty in .env.' } }
  $uri = $null
  if (-not [Uri]::TryCreate($endpoint, [UriKind]::Absolute, [ref]$uri)) { return @{ Ok = $false; Reason = 'ENDPOINT in .env is not a valid URL.' } }
  $version = [regex]::Match($uri.Query, '(?:^\?|&)api-version=([^&]*)')
  $azure = $uri.Scheme -ceq 'https' -and $uri.Host -match '\.openai\.azure\.com\z' -and $uri.AbsolutePath -cmatch '^/openai/deployments/[^/]+/chat/completions/?\z' -and $version.Success
  if (-not $azure) { return @{ Ok = $false; Reason = 'ENDPOINT in .env is not a supported provider URL. Supported: Azure OpenAI chat completions (https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions?api-version=...).' } }
  return @{ Ok = $true; Provider = 'azure-openai'; Key = $key; Uri = $uri; ApiVersion = [Uri]::UnescapeDataString($version.Groups[1].Value) }
}

# Same rules as ChatRequestSchema in src/ai/protocol.ts; returns a problem or $null.
function Get-ChatRequestProblem($req) {
  if ($req -isnot [System.Text.Json.Nodes.JsonObject]) { return 'body: expected an object' }
  $extra = Test-OnlyKeys $req @('messages', 'tools', 'toolChoice', 'maxOutputTokens')
  if ($extra) { return "body: unrecognized key `"$extra`"" }
  $messages = $req['messages']
  if ($messages -isnot [System.Text.Json.Nodes.JsonArray] -or $messages.Count -lt 1 -or $messages.Count -gt $MaxMessages) { return "messages: expected 1 to $MaxMessages messages" }
  $total = 0
  for ($i = 0; $i -lt $messages.Count; $i++) {
    $m = $messages[$i]
    $at = "messages.$i"
    if ($m -isnot [System.Text.Json.Nodes.JsonObject]) { return "${at}: expected an object" }
    $role = Get-JsonString $m['role']
    $allowed = switch -CaseSensitive ($role) {
      'system' { @('role', 'content') }
      'user' { @('role', 'content') }
      'assistant' { @('role', 'content', 'toolCalls') }
      'tool' { @('role', 'toolCallId', 'content') }
      default { $null }
    }
    if ($null -eq $allowed) { return "$at.role: invalid role" }
    $extra = Test-OnlyKeys $m $allowed
    if ($extra) { return "${at}: unrecognized key `"$extra`"" }
    if (-not $m.ContainsKey('content')) { return "$at.content: required" }
    $content = $m['content']
    $text = Get-JsonString $content
    if ($role -ceq 'assistant' -and $null -eq $content) { $text = '' }
    elseif ($null -eq $text) { return "$at.content: expected a string" }
    if ($text.Length -gt $MaxMessageChars) { return "$at.content: longer than $MaxMessageChars characters" }
    if (($role -ceq 'system' -or $role -ceq 'user') -and $text.Length -lt 1) { return "$at.content: must not be empty" }
    $total += $text.Length
    if ($role -ceq 'tool') {
      $id = Get-JsonString $m['toolCallId']
      if ($null -eq $id -or $id -cnotmatch $IdentPattern) { return "$at.toolCallId: invalid" }
    }
    if ($role -ceq 'assistant' -and $m.ContainsKey('toolCalls')) {
      $calls = $m['toolCalls']
      if ($calls -isnot [System.Text.Json.Nodes.JsonArray] -or $calls.Count -gt $MaxToolCalls) { return "$at.toolCalls: expected at most $MaxToolCalls calls" }
      for ($k = 0; $k -lt $calls.Count; $k++) {
        $c = $calls[$k]
        if ($c -isnot [System.Text.Json.Nodes.JsonObject]) { return "$at.toolCalls.${k}: expected an object" }
        $extra = Test-OnlyKeys $c @('id', 'name', 'arguments')
        if ($extra) { return "$at.toolCalls.${k}: unrecognized key `"$extra`"" }
        $id = Get-JsonString $c['id']
        $name = Get-JsonString $c['name']
        $arguments = Get-JsonString $c['arguments']
        if ($null -eq $id -or $id -cnotmatch $IdentPattern) { return "$at.toolCalls.$k.id: invalid" }
        if ($null -eq $name -or $name -cnotmatch $ToolNamePattern) { return "$at.toolCalls.$k.name: invalid" }
        if ($null -eq $arguments -or $arguments.Length -gt $MaxArgumentChars) { return "$at.toolCalls.$k.arguments: expected a string of at most $MaxArgumentChars characters" }
        $total += $arguments.Length
      }
    }
  }
  if ($req.ContainsKey('tools')) {
    $tools = $req['tools']
    if ($tools -isnot [System.Text.Json.Nodes.JsonArray] -or $tools.Count -gt $MaxTools) { return "tools: expected at most $MaxTools tools" }
    for ($k = 0; $k -lt $tools.Count; $k++) {
      $t = $tools[$k]
      if ($t -isnot [System.Text.Json.Nodes.JsonObject]) { return "tools.${k}: expected an object" }
      $extra = Test-OnlyKeys $t @('name', 'description', 'parameters')
      if ($extra) { return "tools.${k}: unrecognized key `"$extra`"" }
      $name = Get-JsonString $t['name']
      $description = Get-JsonString $t['description']
      if ($null -eq $name -or $name -cnotmatch $ToolNamePattern) { return "tools.$k.name: invalid" }
      if ($null -eq $description -or $description.Length -gt $MaxDescriptionChars) { return "tools.$k.description: expected a string of at most $MaxDescriptionChars characters" }
      if ($t['parameters'] -isnot [System.Text.Json.Nodes.JsonObject]) { return "tools.$k.parameters: expected an object" }
    }
    $total += $tools.ToJsonString().Length
  }
  if ($req.ContainsKey('toolChoice')) {
    $choice = Get-JsonString $req['toolChoice']
    if ($choice -cnotin @('auto', 'none', 'required')) { return 'toolChoice: expected auto, none or required' }
  }
  if ($req.ContainsKey('maxOutputTokens')) {
    $n = Get-JsonNumber $req['maxOutputTokens']
    if ($null -eq $n -or $n -ne [Math]::Floor($n) -or $n -lt $MinOutputTokens -or $n -gt $MaxOutputTokens) { return "maxOutputTokens: expected an integer from $MinOutputTokens to $MaxOutputTokens" }
  }
  if ($total -gt $MaxTotalChars) { return "body: the request is longer than $MaxTotalChars characters" }
  return $null
}

# Only MorphLab's own tools may be offered to the model, so the server is not a general relay.
function Get-ForeignTool($req) {
  if (-not $req.ContainsKey('tools')) { return $null }
  foreach ($t in $req['tools']) {
    $name = Get-JsonString $t['name']
    if ($name -cnotin $ToolNames) { return $name }
  }
  return $null
}

# Azure OpenAI chat completions body, as buildAzureBody in server/azure.ts.
# JsonArray and JsonObject values are assigned directly: an if-expression would enumerate them.
function ConvertTo-AzureBody($req, [string]$apiVersion) {
  $messages = [System.Text.Json.Nodes.JsonArray]::new()
  foreach ($m in $req['messages']) {
    $role = Get-JsonString $m['role']
    $o = [System.Text.Json.Nodes.JsonObject]::new()
    $o['role'] = New-JsonValue $role
    switch -CaseSensitive ($role) {
      'assistant' {
        $calls = $null
        if ($m.ContainsKey('toolCalls')) { $calls = $m['toolCalls'] }
        $text = Get-JsonString $m['content']
        if ($null -ne $calls -and $calls.Count -gt 0) {
          $o['content'] = if ($null -eq $text) { $null } else { New-JsonValue $text }
          $mapped = [System.Text.Json.Nodes.JsonArray]::new()
          foreach ($c in $calls) {
            $fn = [System.Text.Json.Nodes.JsonObject]::new()
            $fn['name'] = New-JsonValue (Get-JsonString $c['name'])
            $fn['arguments'] = New-JsonValue (Get-JsonString $c['arguments'])
            $call = [System.Text.Json.Nodes.JsonObject]::new()
            $call['id'] = New-JsonValue (Get-JsonString $c['id'])
            $call['type'] = New-JsonValue 'function'
            $call['function'] = $fn
            $mapped.Add($call)
          }
          $o['tool_calls'] = $mapped
        } else {
          $o['content'] = New-JsonValue $(if ($null -eq $text) { '' } else { $text })
        }
      }
      'tool' {
        $o['tool_call_id'] = New-JsonValue (Get-JsonString $m['toolCallId'])
        $o['content'] = New-JsonValue (Get-JsonString $m['content'])
      }
      default { $o['content'] = New-JsonValue (Get-JsonString $m['content']) }
    }
    $messages.Add($o)
  }
  $body = [System.Text.Json.Nodes.JsonObject]::new()
  $body['messages'] = $messages
  $tools = $null
  if ($req.ContainsKey('tools')) { $tools = $req['tools'] }
  if ($null -ne $tools -and $tools.Count -gt 0) {
    $mapped = [System.Text.Json.Nodes.JsonArray]::new()
    foreach ($t in $tools) {
      $fn = [System.Text.Json.Nodes.JsonObject]::new()
      $fn['name'] = New-JsonValue (Get-JsonString $t['name'])
      $fn['description'] = New-JsonValue (Get-JsonString $t['description'])
      $fn['parameters'] = $t['parameters'].DeepClone()
      $tool = [System.Text.Json.Nodes.JsonObject]::new()
      $tool['type'] = New-JsonValue 'function'
      $tool['function'] = $fn
      $mapped.Add($tool)
    }
    $body['tools'] = $mapped
    $choice = if ($req.ContainsKey('toolChoice')) { Get-JsonString $req['toolChoice'] } else { 'auto' }
    $body['tool_choice'] = New-JsonValue $choice
  }
  $limit = if ($req.ContainsKey('maxOutputTokens')) { [int](Get-JsonNumber $req['maxOutputTokens']) } else { $DefaultOutputTokens }
  $field = if ([string]::CompareOrdinal($apiVersion, '2024-09-01') -ge 0) { 'max_completion_tokens' } else { 'max_tokens' }
  $body[$field] = New-JsonValue $limit
  return $body.ToJsonString()
}

# The proxy's ChatResult from an Azure answer, or $null when the answer has an unexpected shape.
function ConvertFrom-AzureResponse([string]$text) {
  try {
    $j = [System.Text.Json.Nodes.JsonNode]::Parse($text)
    if ($j -isnot [System.Text.Json.Nodes.JsonObject]) { return $null }
    $choices = $j['choices']
    if ($choices -isnot [System.Text.Json.Nodes.JsonArray] -or $choices.Count -lt 1 -or $choices[0] -isnot [System.Text.Json.Nodes.JsonObject]) { return $null }
    $choice = $choices[0]
    $message = $choice['message']
    $out = [System.Text.Json.Nodes.JsonObject]::new()
    $content = if ($message -is [System.Text.Json.Nodes.JsonObject]) { Get-JsonString $message['content'] } else { $null }
    $out['content'] = if ($null -eq $content) { $null } else { New-JsonValue $content }
    $calls = [System.Text.Json.Nodes.JsonArray]::new()
    if ($message -is [System.Text.Json.Nodes.JsonObject] -and $message.ContainsKey('tool_calls') -and $null -ne $message['tool_calls']) {
      if ($message['tool_calls'] -isnot [System.Text.Json.Nodes.JsonArray]) { return $null }
      foreach ($c in $message['tool_calls']) {
        if ($c -isnot [System.Text.Json.Nodes.JsonObject]) { return $null }
        $fn = $c['function']
        if ($fn -isnot [System.Text.Json.Nodes.JsonObject]) { return $null }
        $id = Get-JsonString $c['id']
        $name = Get-JsonString $fn['name']
        $arguments = Get-JsonString $fn['arguments']
        if ($null -eq $id -or $null -eq $name -or $null -eq $arguments) { return $null }
        $call = [System.Text.Json.Nodes.JsonObject]::new()
        $call['id'] = New-JsonValue $id
        $call['name'] = New-JsonValue $name
        $call['arguments'] = New-JsonValue $arguments
        $calls.Add($call)
      }
    }
    $out['toolCalls'] = $calls
    $finish = Get-JsonString $choice['finish_reason']
    $out['finishReason'] = New-JsonValue $(if ($null -eq $finish) { 'unknown' } else { $finish })
    $model = Get-JsonString $j['model']
    $out['model'] = if ($null -eq $model) { $null } else { New-JsonValue $model }
    $usage = $j['usage']
    $prompt = if ($usage -is [System.Text.Json.Nodes.JsonObject]) { Get-JsonNumber $usage['prompt_tokens'] } else { $null }
    $completion = if ($usage -is [System.Text.Json.Nodes.JsonObject]) { Get-JsonNumber $usage['completion_tokens'] } else { $null }
    if ($null -ne $prompt -and $null -ne $completion) {
      $u = [System.Text.Json.Nodes.JsonObject]::new()
      $u['promptTokens'] = New-JsonValue $prompt
      $u['completionTokens'] = New-JsonValue $completion
      $out['usage'] = $u
    } else {
      $out['usage'] = $null
    }
    return $out.ToJsonString()
  } catch {
    return $null
  }
}

# Maps an upstream HTTP error to a safe status and message, as mapHttpError in server/azure.ts.
function Get-UpstreamError([int]$status, [string]$text, $retryAfterSeconds) {
  if ($status -eq 401 -or $status -eq 403) { return @{ Status = 502; Message = 'The AI provider rejected the proxy credentials. Check API_KEY and ENDPOINT in .env.' } }
  if ($status -eq 404) { return @{ Status = 502; Message = 'The AI provider did not find the deployment. Check ENDPOINT in .env.' } }
  if ($status -eq 429) {
    $r = @{ Status = 429; Message = 'The AI provider rate limit was reached. Try again shortly.' }
    if ($null -ne $retryAfterSeconds -and $retryAfterSeconds -gt 0) { $r['RetryAfter'] = [int][Math]::Min([Math]::Ceiling($retryAfterSeconds), 300) }
    return $r
  }
  if ($status -ge 500) { return @{ Status = 502; Message = "The AI provider failed (HTTP $status)." } }
  $code = $null
  $param = $null
  try {
    $e = ([System.Text.Json.Nodes.JsonNode]::Parse($text))['error']
    if ($e -is [System.Text.Json.Nodes.JsonObject]) {
      $code = Get-JsonString $e['code']
      $param = Get-JsonString $e['param']
    }
  } catch { }
  if ($null -ne $code -and $code -cnotmatch $SafeTokenPattern) { $code = $null }
  if ($null -ne $param -and $param -cnotmatch $SafeTokenPattern) { $param = $null }
  if ($code -ceq 'content_filter') { return @{ Status = 422; Message = 'The provider content filter blocked this request.' } }
  $details = @(if ($code) { "code $code" }; if ($param) { "parameter $param" }) -join ', '
  return @{ Status = 502; Message = "The AI provider rejected the request (HTTP $status$(if ($details) { ", $details" }))." }
}

# Maps a URL path to a file under the root, or $null (no traversal, no alternate streams).
function Resolve-StaticPath([string]$rootFull, [string]$urlPath) {
  $rel = [Uri]::UnescapeDataString($urlPath)
  if ($rel -match '[\\:\x00]' -or -not $rel.StartsWith('/')) { return $null }
  if ($rel.EndsWith('/')) { $rel += 'index.html' }
  $segments = $rel.Substring(1).Split('/')
  foreach ($s in $segments) { if ($s -eq '' -or $s -eq '.' -or $s -eq '..') { return $null } }
  $full = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($rootFull, ($segments -join [System.IO.Path]::DirectorySeparatorChar)))
  if (-not $full.StartsWith($rootFull + [System.IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { return $null }
  if (-not [System.IO.File]::Exists($full)) { return $null }
  return $full
}

if ($LibraryOnly) { return }

$RootFull = [System.IO.Path]::GetFullPath($Root).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
if (-not [System.IO.File]::Exists((Join-Path $RootFull 'index.html'))) {
  Write-Host "MorphLab not started: $RootFull has no index.html. Extract dist/ from the CI build artifact (build job, Download artifacts) or pass -Root."
  exit 1
}
$Ai = Read-AiConfig $EnvFile
$AllowedHosts = @("127.0.0.1:$Port", "localhost:$Port")
$AllowedOrigins = @("http://127.0.0.1:$Port", "http://localhost:$Port")
$Sessions = @{}
$Overall = @{ T0 = [Environment]::TickCount64; N = 0 }

function Send-Bytes($ctx, [int]$status, [byte[]]$bytes, [string]$type, [hashtable]$headers = @{}) {
  $res = $ctx.Response
  try {
    $res.StatusCode = $status
    $res.ContentType = $type
    $res.Headers['Cache-Control'] = 'no-store'
    $res.Headers['X-Content-Type-Options'] = 'nosniff'
    foreach ($k in $headers.Keys) { $res.Headers[$k] = [string]$headers[$k] }
    if ($ctx.Request.HttpMethod -cne 'HEAD') {
      $res.ContentLength64 = $bytes.Length
      if ($bytes.Length -gt 0) { $res.OutputStream.Write($bytes, 0, $bytes.Length) }
    }
  } catch {
    # The browser closed the connection.
  } finally {
    try { $res.Close() } catch { }
  }
}

function Send-Text($ctx, [int]$status, [string]$json, [hashtable]$headers = @{}) {
  Send-Bytes $ctx $status ([System.Text.Encoding]::UTF8.GetBytes($json)) 'application/json; charset=utf-8' $headers
}

function Send-Error($ctx, [int]$status, [string]$message, [hashtable]$headers = @{}) {
  $o = [System.Text.Json.Nodes.JsonObject]::new()
  $o['error'] = New-JsonValue $message
  Send-Text $ctx $status $o.ToJsonString() $headers
}

function Write-ApiLog([string]$method, [string]$path, [int]$status, [double]$seconds) {
  Write-Host ('{0:HH:mm:ss} {1} {2} {3} {4:N2} s' -f [DateTime]::Now, $method, $path, $status, $seconds)
}

# Fixed windows per session and overall; returns seconds to wait, or $null when allowed.
function Get-RateLimitWait([string]$session) {
  $t = [Environment]::TickCount64
  if ($t - $Overall['T0'] -ge $WindowMs) { $Overall['T0'] = $t; $Overall['N'] = 0 }
  if ($Sessions.Count -ge 1000) {
    foreach ($id in @($Sessions.Keys)) { if ($t - $Sessions[$id]['T0'] -ge $WindowMs) { $Sessions.Remove($id) } }
  }
  $w = $Sessions[$session]
  if ($null -eq $w -or $t - $w['T0'] -ge $WindowMs) {
    if ($null -eq $w -and $Sessions.Count -ge 1000) { return [int][Math]::Max(1, [Math]::Ceiling(($Overall['T0'] + $WindowMs - $t) / 1000)) }
    $w = @{ T0 = $t; N = 0 }
    $Sessions[$session] = $w
  }
  if ($w['N'] -ge $PerSessionLimit) { return [int][Math]::Max(1, [Math]::Ceiling(($w['T0'] + $WindowMs - $t) / 1000)) }
  if ($Overall['N'] -ge $OverallLimit) { return [int][Math]::Max(1, [Math]::Ceiling(($Overall['T0'] + $WindowMs - $t) / 1000)) }
  $w['N'] = $w['N'] + 1
  $Overall['N'] = $Overall['N'] + 1
  return $null
}

function Read-RequestBody($request) {
  if ($request.ContentLength64 -gt $BodyLimitBytes) { return @{ TooLarge = $true } }
  $buffer = [byte[]]::new(65536)
  $ms = [System.IO.MemoryStream]::new()
  try {
    while (($n = $request.InputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
      $ms.Write($buffer, 0, $n)
      if ($ms.Length -gt $BodyLimitBytes) { return @{ TooLarge = $true } }
    }
    return @{ Text = [System.Text.Encoding]::UTF8.GetString($ms.ToArray()) }
  } finally {
    $ms.Dispose()
  }
}

function Get-StatusJson {
  $o = [System.Text.Json.Nodes.JsonObject]::new()
  if ($Ai.Ok) {
    $o['ok'] = New-JsonValue $true
    $o['provider'] = New-JsonValue $Ai.Provider
    $o['reason'] = $null
    $caps = [System.Text.Json.Nodes.JsonObject]::new()
    foreach ($entry in @(
        @('toolCalling', 'unknown', 'Not probed by this server; depends on the deployed model.'),
        @('imageInput', 'unknown', 'Not probed by this server; depends on the deployed model.'),
        @('streaming', 'no', 'This server returns complete answers.'),
        @('realtimeVoice', 'no', 'A chat completions deployment has no realtime audio; voice uses the browser speech API.'))) {
      $c = [System.Text.Json.Nodes.JsonObject]::new()
      $c['state'] = New-JsonValue $entry[1]
      $c['note'] = New-JsonValue $entry[2]
      $caps[$entry[0]] = $c
    }
    $o['capabilities'] = $caps
  } else {
    $o['ok'] = New-JsonValue $false
    $o['provider'] = $null
    $o['reason'] = New-JsonValue $Ai.Reason
    $o['capabilities'] = $null
  }
  return $o.ToJsonString()
}

# Handles one request; returns a pending upstream call for the event loop, or $null when answered.
function Invoke-Request($ctx) {
  $request = $ctx.Request
  $path = $request.Url.AbsolutePath
  $method = $request.HttpMethod
  $origin = $request.Headers['Origin']
  if ($request.Headers['Host'] -cnotin $AllowedHosts) { Send-Error $ctx 403 'This host is not allowed.'; return $null }
  if ($request.Headers['Sec-Fetch-Site'] -ceq 'cross-site') { Send-Error $ctx 403 'Cross-site requests are not allowed.'; return $null }
  if ($origin -and $origin -cnotin $AllowedOrigins) { Send-Error $ctx 403 'This origin is not allowed.'; return $null }
  if ($method -cnotin @('GET', 'HEAD') -and -not $origin) { Send-Error $ctx 403 'An Origin header is required.'; return $null }

  if ($path -ceq '/api/ai/status' -and $method -ceq 'GET') { Send-Text $ctx 200 (Get-StatusJson); return $null }
  if ($path -ceq '/api/ai/chat' -and $method -ceq 'POST') {
    $started = [DateTime]::UtcNow
    if (-not ([string]$request.ContentType -like 'application/json*')) { Send-Error $ctx 415 'Send JSON with content-type application/json.'; return $null }
    $body = Read-RequestBody $request
    if ($body.ContainsKey('TooLarge')) { Send-Error $ctx 413 'The request body is too large.'; return $null }
    try { $req = [System.Text.Json.Nodes.JsonNode]::Parse($body.Text) } catch { Send-Error $ctx 400 'The request body could not be parsed.'; return $null }
    if (-not $Ai.Ok) { Send-Error $ctx 503 $Ai.Reason; return $null }
    $session = $request.Headers['x-morphlab-session']
    if (-not $session -or $session -cnotmatch $SessionPattern) { Send-Error $ctx 400 'A valid x-morphlab-session header is required.'; return $null }
    $wait = Get-RateLimitWait $session
    if ($null -ne $wait) { Send-Error $ctx 429 'Too many AI requests. Wait a moment and try again.' @{ 'Retry-After' = $wait }; return $null }
    try { $problem = Get-ChatRequestProblem $req } catch { $problem = 'body: invalid structure' }
    if ($problem) { Send-Error $ctx 400 "Invalid chat request: $problem"; return $null }
    $foreign = Get-ForeignTool $req
    if ($null -ne $foreign) { Send-Error $ctx 400 "$foreign is not a MorphLab tool."; return $null }
    $message = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, $Ai.Uri)
    $message.Headers.Add('api-key', $Ai.Key)
    $message.Content = [System.Net.Http.StringContent]::new((ConvertTo-AzureBody $req $Ai.ApiVersion), [System.Text.Encoding]::UTF8, 'application/json')
    $cts = [System.Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds($UpstreamTimeoutSeconds))
    return @{ Ctx = $ctx; Task = $Http.SendAsync($message, $cts.Token); Cts = $cts; Message = $message; Started = $started }
  }
  if ($path.StartsWith('/api/')) { Send-Error $ctx 404 'Not found.'; return $null }
  if ($method -cnotin @('GET', 'HEAD')) { Send-Error $ctx 405 'Method not allowed.'; return $null }
  $file = Resolve-StaticPath $RootFull $path
  if ($null -eq $file) { Send-Bytes $ctx 404 ([System.Text.Encoding]::UTF8.GetBytes('Not found.')) 'text/plain; charset=utf-8'; return $null }
  $type = $MimeTypes[[System.IO.Path]::GetExtension($file).ToLowerInvariant()]
  if (-not $type) { $type = 'application/octet-stream' }
  Send-Bytes $ctx 200 ([System.IO.File]::ReadAllBytes($file)) $type @{ 'Content-Security-Policy' = $Csp }
  return $null
}

function Complete-Chat([hashtable]$p) {
  $ctx = $p.Ctx
  $status = 502
  $response = $null
  try {
    if ($p.Task.IsCanceled -or $p.Task.IsFaulted) {
      if ($p.Cts.IsCancellationRequested) { $status = 504; Send-Error $ctx 504 "The AI provider did not answer within $UpstreamTimeoutSeconds s." }
      else { Send-Error $ctx 502 'The AI provider could not be reached or returned an unreadable answer.' }
      return
    }
    $response = $p.Task.Result
    $code = [int]$response.StatusCode
    $text = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if ($code -ge 200 -and $code -lt 300) {
      $json = ConvertFrom-AzureResponse $text
      if ($null -eq $json) { Send-Error $ctx 502 'The AI provider returned an unexpected response.'; return }
      $status = 200
      Send-Text $ctx 200 $json
      return
    }
    $delta = if ($null -ne $response.Headers.RetryAfter -and $null -ne $response.Headers.RetryAfter.Delta) { $response.Headers.RetryAfter.Delta.TotalSeconds } else { $null }
    $e = Get-UpstreamError $code $text $delta
    $status = $e.Status
    $headers = if ($e.ContainsKey('RetryAfter')) { @{ 'Retry-After' = $e.RetryAfter } } else { @{} }
    Send-Error $ctx $e.Status $e.Message $headers
  } catch {
    Send-Error $ctx 502 'The AI provider could not be reached or returned an unreadable answer.'
  } finally {
    if ($null -ne $response) { $response.Dispose() }
    $p.Message.Dispose()
    $p.Cts.Dispose()
    Write-ApiLog 'POST' '/api/ai/chat' $status ([DateTime]::UtcNow - $p.Started).TotalSeconds
  }
}

$Http = [System.Net.Http.HttpClient]::new()
$Http.Timeout = [System.Threading.Timeout]::InfiniteTimeSpan
$listener = [System.Net.HttpListener]::new()
foreach ($h in $AllowedHosts) { $listener.Prefixes.Add("http://$h/") }
try {
  $listener.Start()
} catch {
  Write-Host "MorphLab not started: cannot listen on 127.0.0.1:$Port ($($_.Exception.GetType().Name))."
  exit 1
}
$aiText = if ($Ai.Ok) { "AI through the local server ($($Ai.Provider))" } else { "AI unavailable: $($Ai.Reason)" }
Write-Host "MorphLab on http://127.0.0.1:$Port/ (app from $RootFull; $aiText). Ctrl+C stops it."
$pending = [System.Collections.Generic.List[hashtable]]::new()
$accept = $listener.GetContextAsync()
try {
  while ($listener.IsListening) {
    $waitOn = [System.Collections.Generic.List[System.Threading.Tasks.Task]]::new()
    $waitOn.Add($accept)
    foreach ($p in $pending) { $waitOn.Add($p.Task) }
    [void][System.Threading.Tasks.Task]::WaitAny($waitOn.ToArray(), 200)
    if ($accept.IsCompleted) {
      if ($accept.IsFaulted -or $accept.IsCanceled) { break }
      $ctx = $accept.Result
      $accept = $listener.GetContextAsync()
      try {
        $next = Invoke-Request $ctx
        if ($null -ne $next) { $pending.Add($next) }
      } catch {
        Send-Error $ctx 500 'Internal server error.'
      }
    }
    for ($i = $pending.Count - 1; $i -ge 0; $i--) {
      if ($pending[$i].Task.IsCompleted) {
        $done = $pending[$i]
        $pending.RemoveAt($i)
        Complete-Chat $done
      }
    }
  }
} finally {
  foreach ($p in $pending) { try { $p.Cts.Cancel() } catch { } }
  $listener.Stop()
  $listener.Close()
  $Http.Dispose()
}
