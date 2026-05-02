FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Create vscode user
ARG USERNAME=vscode
RUN useradd -m -s /bin/bash $USERNAME

# Set the working directory
RUN mkdir /workspace
WORKDIR /workspace

# Install Python, pip, git, curl, wget, gnupg, lsb-release, and other required packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl zsh lsb-release unzip gnupg wget software-properties-common shfmt jq gh

# Install TeX Live (pdflatex) and ChkTeX (lint) – pandoc removed
# Add texlive-extra-utils (provides latexindent) + explicit Perl module deps for full latexindent functionality
RUN apt-get update && apt-get install -y --fix-missing \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    texlive-science \
    texlive-extra-utils \
    chktex \
    latexmk \
    liblog-log4perl-perl \
    libfile-homedir-perl \
    libyaml-tiny-perl \
    python3-pygments && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# Note: texlive-science added to provide mhchem.sty (needed for \usepackage[version=4]{mhchem})

# Switch to vscode user before installing uv
USER $USERNAME

# Ensure ~/.local/bin exists and is in PATH
RUN mkdir -p /home/$USERNAME/.local/bin
ENV PATH="/home/$USERNAME/.local/bin:$PATH"

# Install UV explicitly as vscode user
RUN curl -fsSL https://astral.sh/uv/install.sh | sh

CMD [ "bin/sh" ]