# download base image

FROM "ubuntu:jammy-20231004"


# Install system packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    python3.10 \
    python3-venv \
    python3-pip \
    python3-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*


# add non root user

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# change default terminal to bash (instead of shell)
RUN chsh --shell /bin/bash ${USERNAME}


# Create virtual environment and add to path
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV && chmod -R a+rwx $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install python env requirements
USER vscode
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt

RUN pip3 install torch==2.1.0+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# set vi as git default editor

RUN git config --global core.editor "vim"