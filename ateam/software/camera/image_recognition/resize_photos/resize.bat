@echo off
if exist resized rmdir /s /q resized
mkdir resized
for /f "tokens=*" %%i in ('dir /b *.jpg *.jpeg *.png *.webp *.heic 2^>nul') do (
    ffmpeg -y -i "%%i" -vf scale=640:-2 "resized\%%i"
)
echo Resizing complete! Check resized folder.
pause
