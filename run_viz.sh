#!/usr/bin/env bash

# 用法检查
if [ $# -ne 1 ]; then
    echo "Usage: $0 <middlebury_install_path>"
    exit 1
fi

MIDDLEBURY_PATH="$1"

# 检查路径是否存在
if [ ! -d "$MIDDLEBURY_PATH" ]; then
    echo "Error: directory does not exist: $MIDDLEBURY_PATH"
    exit 1
fi

# 进入目录
cd "$MIDDLEBURY_PATH" || {
    echo "Error: failed to cd into $MIDDLEBURY_PATH"
    exit 1
}

# 检查可执行文件
if [ ! -x "./runviz" ]; then
    echo "Error: ./runviz not found or not executable in $MIDDLEBURY_PATH"
    exit 1
fi

# 执行
echo "Running ./runviz F in $MIDDLEBURY_PATH"
./runviz F
