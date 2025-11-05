# Run programs

## Windows + VSCode

1. Download Visual Studio;
2. Download CUDA Toolkit for Windows from official website;
3. Download NSight Visual Studio Code Edition from VSCode Marketplace;
4. Run program from PowerShell using:
   ```powershell
   nvcc your_program.cu -o your_program.exe -ccbin path\to\cl.exe
   ```
   For Visual Studio 2022 the path to cl.exe is: `"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"`.
   
> [!TIP]
>
> You can avoid specify every time `-ccbin` adding the path to cl.exe to PATH environment variable. In order to do this, press `Win+R`, type `sysdm.cpl`, go to "Advanced", click on "Environment Variables..." and edit `Path`.