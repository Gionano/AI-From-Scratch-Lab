$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonPath = $venvPython
} else {
    $pythonPath = Join-Path $env:LOCALAPPDATA "Programs\Python\Python314\python.exe"
    if (-not (Test-Path $pythonPath)) {
        $pythonPath = "python"
    }
}

& $pythonPath "show_ai_stack.py" @args
