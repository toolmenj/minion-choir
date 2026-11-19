@echo off
rem start local server and open the page (close this window to stop)
cd /d "%~dp0"
start "" http://127.0.0.1:8765/index.html
python -m http.server 8765 --bind 127.0.0.1
