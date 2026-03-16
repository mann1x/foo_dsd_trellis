@echo off
REM Setup conda environment for foo_dsd_trellis ML training pipeline
REM Usage: setup_env.bat

echo Creating conda environment 'foo_dsd_trellis'...
call conda create -n foo_dsd_trellis python=3.11 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment
    exit /b 1
)

echo Activating environment...
call conda activate foo_dsd_trellis

echo Installing PyTorch nightly with CUDA 12.8 (Blackwell/sm_120 support)...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
if errorlevel 1 (
    echo WARNING: Nightly install failed, trying stable CUDA 12.4...
    call conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
)

echo Installing additional packages...
pip install ^
    numpy ^
    scipy ^
    soundfile ^
    librosa ^
    onnx ^
    onnxruntime ^
    onnxruntime-directml ^
    matplotlib ^
    tqdm ^
    tensorboard

echo.
echo Environment 'foo_dsd_trellis' is ready.
echo Activate with: conda activate foo_dsd_trellis
echo.
echo To verify: python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
