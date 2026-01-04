#!/bin/bash -e
# -*- coding: utf-8 -*-

wget -P datasets https://huggingface.co/datasets/uva-cv-lab/ovmono3d_data/resolve/main/ovmono3d_data.zip
unzip datasets/ovmono3d_data.zip -d datasets/Omni3D

