#!/usr/bin/env bash
# Launch the El Nino - Sri Lanka dashboard in the dedicated conda env.
set -euo pipefail
cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate elnino-lk
exec streamlit run app.py "$@"
