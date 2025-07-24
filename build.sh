#!/usr/bin/env bash

# Python asılılıqlarını quraşdırın
pip install -r requirements.txt

# spaCy modelini yükləyin
python3 -m spacy download en_core_web_sm

echo "Build script tamamlandı."