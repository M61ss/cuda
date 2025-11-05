# Run programs

## Windows + VSCode

1. Download and install Visual Studio;
2. Download and install CUDA Toolkit for Windows from official website;
3. Download and install NSight Visual Studio Code Edition from VSCode Marketplace;
4. Run program from PowerShell using:
   ```powershell
   nvcc your_program.cu -o your_program.exe -ccbin path\to\cl.exe
   ```
   For Visual Studio 2022 the path to cl.exe is: `"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"`.
   
> [!TIP]
>
> You can avoid specify every time `-ccbin` adding the path to cl.exe to PATH environment variable. In order to do this, press `Win+R`, type `sysdm.cpl`, go to "Advanced", click on "Environment Variables..." and edit `Path`.

# Profiling

## Windows + NSight CLI

1. If not installed, download and install NSight System (its stable version should be bundled with CUDA Toolkit). To check if it is installed, look at `"C:\Program Files\NVIDIA Corporation\"`;
2. Add nsys.exe to the path (for instance its path could be `"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.5.1\target-windows-x64\"`). 