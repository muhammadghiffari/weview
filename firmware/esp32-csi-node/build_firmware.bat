@echo off
echo STARTING > C:\Users\ruv\idf_test.txt
set IDF_PATH=C:\Users\ruv\esp\v5.4\esp-idf
set PATH=C:\Espressif\tools\python\v5.4\venv\Scripts;C:\Espressif\tools\xtensa-esp-elf\esp-14.2.0_20241119\xtensa-esp-elf\bin;C:\Espressif\tools\cmake\3.30.2\bin;C:\Espressif\tools\ninja\1.12.1;C:\Espressif\tools\idf-exe\1.0.3;%PATH%
echo PATH_SET >> C:\Users\ruv\idf_test.txt
cd /d C:\Users\ruv\Projects\wifi-densepose\firmware\esp32-csi-node
echo CD_DONE >> C:\Users\ruv\idf_test.txt
python %IDF_PATH%\tools\idf.py build >> C:\Users\ruv\idf_test.txt 2>&1
echo RC=%ERRORLEVEL% >> C:\Users\ruv\idf_test.txt
