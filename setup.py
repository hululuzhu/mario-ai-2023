from setuptools import setup, find_packages

setup(
    name = 'mario-ai-2023',
    version = '0.1.0',
    url = 'https://github.com/hululuzhu/mario-ai-2023',
    description = 'A fun AI intro course to train Super Mario AI',
    packages = find_packages(),
    install_requires = [
        'ffmpeg-python',
        'gym-super-mario-bros',
        'gym==0.22',
        'PyOpenGL',
        'pyvirtualdisplay',
        'stable-baselines3[extra]==1.0.0',
        'xvfbwrapper',
    ]
)
