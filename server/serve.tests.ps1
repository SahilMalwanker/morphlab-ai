#Requires -Version 7.3
# Tests for server/serve.ps1 (local, PowerShell only). Fake key and endpoint unless -Live is given;
# -Live adds one real request through the server and prints only status codes and answer shapes.
# Usage: pwsh -NoProfile -File server/serve.tests.ps1 [-Live]
param([switch]$Live)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'serve.ps1') -LibraryOnly
Set-StrictMode -Off

$script:failures = 0
$script:checks = 0
function Check([string]$name, [bool]$ok) {
  $script:checks++
  if ($ok) { Write-Host "PASS $name" } else { $script:failures++; Write-Host "FAIL $name" }
}

$fakeKey = 'fake-key-0123456789abcdef'
$fakeHost = 'fake-resource.openai.azure.com'
$fakeEndpoint = "https://$fakeHost/openai/deployments/fake-model/chat/completions?api-version=2024-10-21"
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("morphlab-serve-tests-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null

function Parse([string]$json) { return , [System.Text.Json.Nodes.JsonNode]::Parse($json) }

try {
  # Configuration
  $envFile = Join-Path $tmp '.env'
  Check 'config: missing file' ((Read-AiConfig (Join-Path $tmp 'none.env')).Reason -like 'The .env file was not found*')
  Set-Content -LiteralPath $envFile -Value ('ENDPOINT="' + $fakeEndpoint + '"')
  $c = Read-AiConfig $envFile
  Check 'config: missing API_KEY named' (-not $c.Ok -and $c.Reason -ceq 'API_KEY is missing or empty in .env.')
  Set-Content -LiteralPath $envFile -Value @(('API_KEY="' + $fakeKey + '"'), ('ENDPOINT="https://' + $fakeHost + '.attacker.example/openai/deployments/x/chat/completions?api-version=1"'))
  $c = Read-AiConfig $envFile
  Check 'config: foreign host refused without echoing it' (-not $c.Ok -and $c.Reason -like '*not a supported provider URL*' -and $c.Reason -notlike "*$fakeHost*" -and $c.Reason -notlike "*$fakeKey*")
  Set-Content -LiteralPath $envFile -Value @('# comment', ('API_KEY="' + $fakeKey + '"'), ('ENDPOINT="' + $fakeEndpoint + '"'))
  $c = Read-AiConfig $envFile
  Check 'config: Azure OpenAI deployment accepted' ($c.Ok -and $c.Provider -ceq 'azure-openai' -and $c.ApiVersion -ceq '2024-10-21')

  # Request validation (same rules as ChatRequestSchema)
  $user = '{"role":"user","content":"Drive to the red cube."}'
  $userOnly = '{"messages":[' + $user + ']}'
  Check 'validate: minimal request' ($null -eq (Get-ChatRequestProblem (Parse $userOnly)))
  $withUser = { param([string]$rest) '{"messages":[' + $user + '],' + $rest + '}' }
  $longUser = '{"role":"user","content":"' + ('x' * 7000) + '"}'
  $bad = @{
    'extra top-level key' = & $withUser '"temperature":2'
    'unknown role' = '{"messages":[{"role":"root","content":"x"}]}'
    'empty user content' = '{"messages":[{"role":"user","content":""}]}'
    'no messages' = '{"messages":[]}'
    'tool without valid id' = '{"messages":[{"role":"tool","toolCallId":"bad id","content":"{}"}]}'
    'tool call name with spaces' = '{"messages":[{"role":"assistant","content":null,"toolCalls":[{"id":"c1","name":"set joint","arguments":"{}"}]}]}'
    'toolChoice outside the enum' = & $withUser '"toolChoice":"any"'
    'maxOutputTokens too small' = & $withUser '"maxOutputTokens":15'
    'maxOutputTokens not an integer' = & $withUser '"maxOutputTokens":16.5'
    'tool parameters not an object' = & $withUser '"tools":[{"name":"navigateTo","description":"d","parameters":[]}]'
    'message too long' = '{"messages":[{"role":"user","content":"' + ('x' * 16001) + '"}]}'
    'request too long in total' = '{"messages":[' + ((1..7 | ForEach-Object { $longUser }) -join ',') + ']}'
  }
  foreach ($k in $bad.Keys) { Check "validate: refuses $k" ($null -ne (Get-ChatRequestProblem (Parse $bad[$k]))) }
  $many = '{"messages":[' + ((1..41 | ForEach-Object { $user }) -join ',') + ']}'
  Check 'validate: refuses 41 messages' ($null -ne (Get-ChatRequestProblem (Parse $many)))
  $img = 'data:image/jpeg;base64,/9j/4AAQSkZJRg=='
  $imgBad = @{
    'an image on a user message (D31)' = '{"messages":[{"role":"user","content":"Camera image.","image":"' + $img + '"}]}'
    'an image on a system message' = '{"messages":[{"role":"system","content":"x","image":"' + $img + '"}]}'
  }
  foreach ($k in $imgBad.Keys) { Check "validate: refuses $k" ($null -ne (Get-ChatRequestProblem (Parse $imgBad[$k]))) }
  $assistant = '{"messages":[{"role":"user","content":"Go."},{"role":"assistant","content":null,"toolCalls":[{"id":"call_0","name":"getRobotState","arguments":"{}"}]},{"role":"tool","toolCallId":"call_0","content":"{\"status\":\"succeeded\"}"}],"tools":[{"name":"navigateTo","description":"Drive.","parameters":{"type":"object","properties":{}}}],"maxOutputTokens":256}'
  Check 'validate: tool-call conversation' ($null -eq (Get-ChatRequestProblem (Parse $assistant)))
  $foreignTool = & $withUser '"tools":[{"name":"shell_exec","description":"Run.","parameters":{}}]'
  Check 'allowlist: foreign tool named' ((Get-ForeignTool (Parse $foreignTool)) -ceq 'shell_exec')
  Check 'allowlist: MorphLab tool passes' ($null -eq (Get-ForeignTool (Parse $assistant)))

  # Azure request mapping (as in tests/server/app.test.ts)
  $expected = '{"messages":[{"role":"user","content":"Go."},{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"getRobotState","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_0","content":"{\u0022status\u0022:\u0022succeeded\u0022}"}],"tools":[{"type":"function","function":{"name":"navigateTo","description":"Drive.","parameters":{"type":"object","properties":{}}}}],"tool_choice":"auto","max_completion_tokens":256}'
  Check 'azure body: messages, tools, tool_choice, max_completion_tokens' ((ConvertTo-AzureBody (Parse $assistant) '2024-10-21') -ceq $expected)
  $older = ConvertTo-AzureBody (Parse $userOnly) '2024-06-01'
  Check 'azure body: max_tokens before 2024-09-01, no tools' ($older -ceq '{"messages":[{"role":"user","content":"Drive to the red cube."}],"max_tokens":1024}')

  # Azure response mapping
  $answer = '{"model":"fake-model-2024","choices":[{"finish_reason":"tool_calls","message":{"content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"navigateTo","arguments":"{\"x\":1,\"y\":0}"}}]}}],"usage":{"prompt_tokens":120,"completion_tokens":18,"total_tokens":138}}'
  Check 'azure answer: tool calls and usage mapped' ((ConvertFrom-AzureResponse $answer) -ceq '{"content":null,"toolCalls":[{"id":"call_1","name":"navigateTo","arguments":"{\u0022x\u0022:1,\u0022y\u0022:0}"}],"finishReason":"tool_calls","model":"fake-model-2024","usage":{"promptTokens":120,"completionTokens":18}}')
  Check 'azure answer: plain text' ((ConvertFrom-AzureResponse '{"choices":[{"finish_reason":"stop","message":{"content":"Hi."}}]}') -ceq '{"content":"Hi.","toolCalls":[],"finishReason":"stop","model":null,"usage":null}')
  Check 'azure answer: unexpected shapes refused' ($null -eq (ConvertFrom-AzureResponse '{"unexpected":true}') -and $null -eq (ConvertFrom-AzureResponse 'not json') -and $null -eq (ConvertFrom-AzureResponse '{"choices":[{"message":{"tool_calls":[{"id":"a"}]}}]}'))

  # Upstream errors: fixed messages, never the upstream body
  $leaky = '{"error":{"code":"BadRequest","message":"deployment fake-model at ' + $fakeHost + ' rejected key ' + $fakeKey + '","param":"max_completion_tokens"}}'
  $e = Get-UpstreamError 400 $leaky $null
  Check 'upstream 400: code and parameter only' ($e.Status -eq 502 -and $e.Message -ceq 'The AI provider rejected the request (HTTP 400, code BadRequest, parameter max_completion_tokens).')
  Check 'upstream 401: credentials message' ((Get-UpstreamError 401 "bad key $fakeKey" $null).Message -ceq 'The AI provider rejected the proxy credentials. Check API_KEY and ENDPOINT in .env.')
  Check 'upstream 404: deployment message' ((Get-UpstreamError 404 '{}' $null).Status -eq 502)
  $r = Get-UpstreamError 429 '{}' 12
  Check 'upstream 429: Retry-After passed on' ($r.Status -eq 429 -and $r.RetryAfter -eq 12)
  Check 'upstream content filter: 422' ((Get-UpstreamError 400 '{"error":{"code":"content_filter","message":"x"}}' $null).Status -eq 422)
  Check 'upstream 503: 502 with status' ((Get-UpstreamError 503 "$fakeHost overloaded" $null).Message -ceq 'The AI provider failed (HTTP 503).')

  # Static paths
  $root = Join-Path $tmp 'dist'
  New-Item -ItemType Directory -Path (Join-Path $root 'assets') -Force | Out-Null
  Set-Content -LiteralPath (Join-Path $root 'index.html') -Value '<!doctype html><title>MorphLab test</title>'
  [System.IO.File]::WriteAllBytes((Join-Path $root 'assets\m.wasm'), [byte[]](0, 97, 115, 109))
  Set-Content -LiteralPath (Join-Path $root 'assets\app.js') -Value 'export {};'
  Copy-Item -LiteralPath $envFile -Destination (Join-Path $tmp 'secret.txt')
  $rootFull = [System.IO.Path]::GetFullPath($root)
  Check 'static: / maps to index.html' ((Resolve-StaticPath $rootFull '/') -ceq (Join-Path $rootFull 'index.html'))
  Check 'static: asset resolves' ((Resolve-StaticPath $rootFull '/assets/m.wasm') -ceq (Join-Path $rootFull 'assets\m.wasm'))
  foreach ($p in '/../secret.txt', '/%2e%2e/secret.txt', '/assets/..%2f..%2fsecret.txt', '/assets/m.wasm:stream', '/assets\..\..\secret.txt', '/missing.js', '/assets/') {
    Check "static: refuses $p" ($null -eq (Resolve-StaticPath $rootFull $p))
  }

  # HTTP: a child server with the fake configuration
  $port = 47173
  $base = "http://127.0.0.1:$port"
  $pwsh = (Get-Process -Id $PID).Path
  $script = Join-Path $PSScriptRoot 'serve.ps1'
  $server = Start-Process -FilePath $pwsh -ArgumentList @('-NoProfile', '-File', "`"$script`"", '-Port', $port, '-Root', "`"$root`"", '-EnvFile', "`"$envFile`"") -PassThru -NoNewWindow -RedirectStandardOutput (Join-Path $tmp 'out.txt') -RedirectStandardError (Join-Path $tmp 'err.txt')
  $client = [System.Net.Http.HttpClient]::new()
  $client.Timeout = [TimeSpan]::FromSeconds(20)
  function Send([string]$method, [string]$path, [string]$body, [hashtable]$headers = @{}, [string]$type = 'application/json') {
    $m = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::new($method), "$base$path")
    foreach ($k in $headers.Keys) { if ($k -ceq 'Host') { $m.Headers.Host = $headers[$k] } else { [void]$m.Headers.TryAddWithoutValidation($k, [string]$headers[$k]) } }
    if ($null -ne $body) { $m.Content = [System.Net.Http.StringContent]::new($body, [System.Text.Encoding]::UTF8, $type) }
    $res = $client.SendAsync($m).GetAwaiter().GetResult()
    $text = $res.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    $h = @{}
    foreach ($kv in $res.Headers) { $h[$kv.Key] = ($kv.Value -join ',') }
    foreach ($kv in $res.Content.Headers) { $h[$kv.Key] = ($kv.Value -join ',') }
    return @{ Status = [int]$res.StatusCode; Text = $text; Headers = $h }
  }
  $up = $false
  for ($i = 0; $i -lt 50 -and -not $up; $i++) {
    try { $up = (Send 'GET' '/api/ai/status' $null).Status -eq 200 } catch { Start-Sleep -Milliseconds 200 }
  }
  Check 'http: server started' $up
  if ($up) {
    $origin = @{ Origin = $base; 'x-morphlab-session' = 'session-test-0001' }
    $all = [System.Collections.Generic.List[string]]::new()
    $s = Send 'GET' '/' $null
    $all.Add($s.Text)
    Check 'http: index.html with CSP and nosniff' ($s.Status -eq 200 -and $s.Headers['Content-Type'] -like 'text/html*' -and $s.Headers['Content-Security-Policy'] -like "*connect-src 'self'*" -and $s.Headers['X-Content-Type-Options'] -ceq 'nosniff')
    Check 'http: wasm MIME type' ((Send 'GET' '/assets/m.wasm' $null).Headers['Content-Type'] -ceq 'application/wasm')
    Check 'http: traversal is 404' ((Send 'GET' '/%2e%2e/secret.txt' $null).Status -eq 404)
    $s = Send 'GET' '/api/ai/status' $null
    $all.Add($s.Text)
    $status = $s.Text | ConvertFrom-Json
    Check 'http: status reports the provider' ($status.ok -eq $true -and $status.provider -ceq 'azure-openai' -and $status.capabilities.realtimeVoice.state -ceq 'no')
    Check 'http: unknown API route is 404 JSON' ((Send 'GET' '/api/other' $null).Text -ceq '{"error":"Not found."}')
    $chat = $userOnly
    Check 'http: POST without Origin refused' ((Send 'POST' '/api/ai/chat' $chat @{ 'x-morphlab-session' = 'session-test-0001' }).Status -eq 403)
    Check 'http: foreign Origin refused' ((Send 'POST' '/api/ai/chat' $chat @{ Origin = 'https://evil.example'; 'x-morphlab-session' = 'session-test-0001' }).Status -eq 403)
    Check 'http: cross-site refused' ((Send 'POST' '/api/ai/chat' $chat ($origin + @{ 'Sec-Fetch-Site' = 'cross-site' })).Status -eq 403)
    Check 'http: foreign Host refused' ((Send 'GET' '/api/ai/status' $null @{ Host = "evil.example:$port" }).Status -in 400, 403)
    Check 'http: text/plain is 415' ((Send 'POST' '/api/ai/chat' 'hello' $origin 'text/plain').Status -eq 415)
    Check 'http: broken JSON is 400' ((Send 'POST' '/api/ai/chat' '{"messages":' $origin).Text -ceq '{"error":"The request body could not be parsed."}')
    Check 'http: oversized body is 413' ((Send 'POST' '/api/ai/chat' ('{"messages":[{"role":"user","content":"' + ('x' * 300000) + '"}]}') $origin).Status -eq 413)
    Check 'http: session header required' ((Send 'POST' '/api/ai/chat' $chat @{ Origin = $base; 'x-morphlab-session' = 'short' }).Status -eq 400)
    $s = Send 'POST' '/api/ai/chat' $foreignTool $origin
    $all.Add($s.Text)
    Check 'http: foreign tool refused' ($s.Status -eq 400 -and $s.Text -ceq '{"error":"shell_exec is not a MorphLab tool."}')
    $limited = @{ Origin = $base; 'x-morphlab-session' = 'session-test-0002' }
    $codes = 1..20 | ForEach-Object { (Send 'POST' '/api/ai/chat' '{"messages":[]}' $limited).Status }
    $s = Send 'POST' '/api/ai/chat' '{"messages":[]}' $limited
    Check 'http: 20 requests per session, then 429 with Retry-After' (@($codes | Where-Object { $_ -ne 400 }).Count -eq 0 -and $s.Status -eq 429 -and [int]$s.Headers['Retry-After'] -ge 1)
    Check 'http: PUT with Origin is 405' ((Send 'PUT' '/' 'x' $origin).Status -eq 405)
    Check 'http: no response contains the key or the endpoint host' (-not ($all | Where-Object { $_ -like "*$fakeKey*" -or $_ -like "*$fakeHost*" }))
  }
  $client.Dispose()
  if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
  $log = (Get-Content -Raw -LiteralPath (Join-Path $tmp 'out.txt') -ErrorAction SilentlyContinue) + (Get-Content -Raw -LiteralPath (Join-Path $tmp 'err.txt') -ErrorAction SilentlyContinue)
  Check 'http: server output has no key or endpoint host' (-not ([string]$log -like "*$fakeKey*" -or [string]$log -like "*$fakeHost*"))

  if ($Live) {
    # One real request through the server with the real .env; output limited to status and shape.
    $port = 47174
    $base = "http://127.0.0.1:$port"
    $liveServer = Start-Process -FilePath $pwsh -ArgumentList @('-NoProfile', '-File', "`"$script`"", '-Port', $port, '-Root', "`"$root`"") -PassThru -NoNewWindow -RedirectStandardOutput (Join-Path $tmp 'live-out.txt') -RedirectStandardError (Join-Path $tmp 'live-err.txt')
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds(60)
    $up = $false
    for ($i = 0; $i -lt 50 -and -not $up; $i++) {
      try { $up = (Send 'GET' '/api/ai/status' $null).Status -eq 200 } catch { Start-Sleep -Milliseconds 200 }
    }
    $headers = @{ Origin = $base; 'x-morphlab-session' = 'session-live-0001' }
    $tool = '{"name":"navigateTo","description":"Drive the rover to a goal on the arena floor (world frame, m).","parameters":{"type":"object","properties":{"x":{"type":"number","minimum":-0.4,"maximum":3.4},"y":{"type":"number","minimum":-1.9,"maximum":1.9}},"required":["x","y"],"additionalProperties":false}}'
    $liveBody = '{"messages":[{"role":"system","content":"You operate a simulated rover only through tools."},{"role":"user","content":"Drive to x 1 m, y 0.5 m."}],"tools":[' + $tool + '],"maxOutputTokens":200}'
    $s = Send 'POST' '/api/ai/chat' $liveBody $headers
    $shape = 'no answer'
    if ($s.Status -eq 200) {
      $j = $s.Text | ConvertFrom-Json
      $shape = "finishReason $($j.finishReason), tool calls: $((@($j.toolCalls) | ForEach-Object { $_.name }) -join ', '), model $($j.model), tokens $($j.usage.promptTokens)/$($j.usage.completionTokens)"
    }
    Write-Host "LIVE status $($s.Status): $shape"
    Check 'live: model answered through the server with a tool call' ($s.Status -eq 200 -and @(($s.Text | ConvertFrom-Json).toolCalls).Count -ge 1)
    $client.Dispose()
    if (-not $liveServer.HasExited) { Stop-Process -Id $liveServer.Id -Force }
  }
} finally {
  Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
Write-Host "$($script:checks - $script:failures) of $($script:checks) checks passed"
if ($script:failures -gt 0) { exit 1 }
