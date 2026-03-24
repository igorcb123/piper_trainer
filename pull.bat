@echo off
REM Pull latest changes from origin main.
REM Usage: double-click or run from project root in PowerShell/CMD.

cd /d "%~dp0"
echo Fetching and pulling origin/main...
git fetch origin
git checkout main 2>NUL || echo Already on main or branch missing.
git pull origin main
echo Done.
