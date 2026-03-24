@echo off
REM Push all changes to origin main with a generic commit message.
REM Usage: double-click or run from project root in PowerShell/CMD.

cd /d "%~dp0"
echo Adding all changes...
git add -A
echo Committing (if there are changes)...
git commit -m "Automated commit: update project" 2>NUL || echo No changes to commit.
echo Pushing to origin/main...
git push origin main
echo Done.
