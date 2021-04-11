@ECHO OFF

if "%1" == "" goto help

if "%1" == "help" (
    :help
    echo.
    echo.Please use `make ^<target^>` where ^<target^> is one of
    echo. conda-update: Create or update conda env
    echo. conda-remove: Remove conda env
    echo.
)

@REM Create or update conda env
if "%1" == "conda-update" (
    conda env update --prune --file environment.yml
    conda activate im2latex
    conda develop .
    goto end
)

@REM Remove conda env
if "%1" == "conda-remove" (
    conda deactivate
    conda env remove -n im2latex
    goto end
)


:end