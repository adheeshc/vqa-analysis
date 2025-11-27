#!/bin/bash

# Initialize and configure git submodules with sparse checkout

echo "Initializing submodules..."
git submodule init
git submodule update --init --remote --depth 1 models/vlm_chatbot

echo "Configuring sparse checkout..."
git config -f .git/modules/models/vlm_chatbot/config core.sparseCheckout true
mkdir -p .git/modules/models/vlm_chatbot/info
echo "models/*" > .git/modules/models/vlm_chatbot/info/sparse-checkout

cd models/vlm_chatbot
git read-tree -mu HEAD
cd ../..

echo "Done! Submodules initialized with sparse checkout."
