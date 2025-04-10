"""
Make sure there are `dir_name/__init__.py` and `dir_name/__main__.py` under dir_name/

pip install -e .       # 安装包 (-e: 开发模式,代码修改实时生效)
pip uninstall command  # 卸载包

After setup, you can replace
*    `python -m dir_name` -> `command`
"""
from setuptools import setup, find_packages


dir_name = "titanic"
command = "titanic"
main_function = "cli"  # main function of dir_name/__main__.py

version = "0.2"
python_requires = ">=3.11.0,<3.13.0"  # python 3.11/3.12
install_requires = [
    "click==8.1.8",
    "pandas==2.2.3",
    "scikit-learn==1.6.1",

    # pytorch 2.6 (CPU)
    "torch>=2.6.0,<2.7",
    "torchaudio>=2.6.0,<2.7",
    "torchvision>=0.21.0,<0.22",
]


setup(
    name=dir_name,
    version=version,
    packages=[dir_name],  # packages=find_packages(),
    python_requires=python_requires,
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            # 将 command 映射到 __main__.py 的 main_function 函数
            f"{command} = {dir_name}.__main__:{main_function}",
        ],
    },
)