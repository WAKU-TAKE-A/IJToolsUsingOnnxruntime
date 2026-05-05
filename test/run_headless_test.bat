@echo off
setlocal

:: Path to the project root
set PROJECT_ROOT=C:\Users\admin\Documents\NetBeansProjects\IJToolsUsingOnnxruntime

:: Use JAVA_HOME if set, otherwise assume java is in PATH
if defined JAVA_HOME (
    set JAVA_CMD="%JAVA_HOME%\bin\java.exe"
) else (
    set JAVA_CMD=java
)

echo Starting ImageJ Headless Test...
cd /d "%PROJECT_ROOT%"

:: Run ImageJ with the macro
:: -batch: Headless mode
:: -macro: Path to the .ijm script
%JAVA_CMD% -Xmx2g -jar ij.jar -batch test/test_all_models.ijm

echo.
echo Test Execution Finished.
echo Check "test/test_log.txt" for details.
pause
