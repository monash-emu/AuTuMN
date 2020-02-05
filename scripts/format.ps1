# Use black to format the codebase
python -m black `
    --line-length 100 `
    --include autumn/**/*.py applications/**/*.py tests/**/*.py `
    .
