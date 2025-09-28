from setuptools import setup, find_packages

setup(
name="gpt_mini8_sdk",  # Nome da sua biblioteca
version="1.0.0",
author="Augusto Miguel de Farias",
author_email="augustomiguelfarias7@gmail.com",
description="GPT-Mini 8 SDK: Biblioteca multimodal Python para GPT-Mini 8, com suporte a texto, COT, imagens e servidor Flask.",
long_description=open("README.md", encoding="utf-8").read(),
long_description_content_type="text/markdown",
url="https://github.com/augustomiguelfarias7-cmd/gpt_mini8_sdk",
packages=find_packages(),
classifiers=[
"Programming Language :: Python :: 3",
"License :: OSI Approved :: Apache Software License",
"Operating System :: OS Independent",
],
python_requires=">=3.11",
install_requires=[
"torch>=2.0.0",
"tokenizers>=0.13.3",
"flask>=2.3.0",
"requests>=2.31.0",
"pillow>=10.0.0",
"numpy>=1.26.0",
"torchvision>=0.16.0"
],
entry_points={
"console_scripts": [
# Caso queira criar um comando direto no terminal
"gpt-mini8=gpt_mini8_sdk.cli:main",
],
},
)

