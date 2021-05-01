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

@REM Build api-server docker image
if "%1" == "build-image" (
    docker rmi -f im2latex/api-server
    docker build -t im2latex/api-server -f api_server/Dockerfile .
    goto end
)

@REM Build api-server docker image
if "%1" == "run-container" (
    docker rm -f im2latex-api
    docker run -p 8080:8000 -it --rm --name im2latex-api im2latex/api-server
    goto end
)

:end