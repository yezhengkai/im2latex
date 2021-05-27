@ECHO OFF

if "%1" == "" goto help

if "%1" == "help" (
    :help
    echo.
    echo.Please use `make ^<target^>` where ^<target^> is one of
    echo. conda-update: Create or update conda env
    echo. conda-remove: Remove conda env
    echo. build-image: Build im2latex/api-server docker image
    echo. run-container: Run im2latex-api container
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

@REM Build im2latex/api-server docker image
if "%1" == "build-image" (
    docker rmi -f im2latex/api-server
    docker build -t im2latex/api-server -f api_server/Dockerfile .
    goto end
)

@REM Run im2latex-api container
if "%1" == "run-container" (
    docker rm -f im2latex-api
    docker run -p 60000:60000 -p 60001:60001 -it --rm --name im2latex-api im2latex/api-server
    goto end
)

:end