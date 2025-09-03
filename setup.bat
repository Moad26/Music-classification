@echo off
setlocal enabledelayedexpansion

:: Create directories
if not exist model mkdir model
if not exist input mkdir input

:: Ask about keeping zip files
:ask_confirm
set /p conf="Do u wanna leave the .zip files [y/n]: "
if /i "%conf%"=="y" goto confirm_ok
if /i "%conf%"=="n" goto confirm_ok
echo just [y/n] don't waste time
goto ask_confirm

:confirm_ok
cd input

:: Download files (using curl if available, otherwise powershell)
where curl >nul 2>nul
if !errorlevel! equ 0 (
    curl -O https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip
    curl -O https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip
    curl -O https://cvml.unige.ch/databases/DEAM/features.zip
) else (
    powershell -Command "Invoke-WebRequest -Uri 'https://cvml.unige.ch/databases/DEAM/DEAM_Annotations.zip' -OutFile 'DEAM_Annotations.zip'"
    powershell -Command "Invoke-WebRequest -Uri 'https://cvml.unige.ch/databases/DEAM/DEAM_audio.zip' -OutFile 'DEAM_audio.zip'"
    powershell -Command "Invoke-WebRequest -Uri 'https://cvml.unige.ch/databases/DEAM/features.zip' -OutFile 'features.zip'"
)

:: Process each file
for %%f in (DEAM_Annotations.zip DEAM_audio.zip features.zip) do (
    if exist "%%f" (
        powershell -Command "Expand-Archive -Path '%%f' -DestinationPath '.' -Force"
        if !errorlevel! neq 0 echo %%f is corrupted
        if /i "%conf%"=="n" del "%%f"
    )
)

cd ..
echo Download complete!
pause
