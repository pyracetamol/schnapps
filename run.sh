#!/bin/env bash

declare -A apps=(["open_day_allen-cahn_app.py"]="8501" ["open_day_cahn-hilliard_app.py"]="8502" ["open_day_mc-app.py"]="8503")

for app in "${!apps[@]}"
do
    streamlit run --server.port "${apps[$app]}" $app &
done
