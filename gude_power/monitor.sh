#!/bin/bash

USER="admin"
PASSWORD="StU_dent#20.25"
HOST="powermeter04.cit.tu-berlin.de"
OUTPUT_IDLE="output_idle.json"
OUTPUT_1="output1.json"
OUTPUT_2="output2.json"
OUTPUT_3="output3.json"
COMPONENTS="8470528"
RUNTIME=60 # in seconds
START_TIME=$SECONDS

echo "Cleaning up old output file..."
if [ -f "$OUTPUT_FILE" ]; then
    echo "File \"$OUTPUT_FILE\" exists, removing it."
    rm "$OUTPUT_FILE"
fi

echo "Starting power monitoring..."

# TODO:
# 1. capture start energy

# TODO: Add a time condition to the loop
while [ $((SECONDS - START_TIME)) -lt $RUNTIME ]; do
    curl -k -u "$USER:$PASSWORD" "https://$HOST/status.json?components=$COMPONENTS" | jq '
    . as $r
    | ($r.sensor_descr[] | select(.type==8)) as $D
    | ($r.sensor_values[] | select(.type==8)) as $V
    | [ range(0; $D.num) as $i
        | { name: $D.properties[$i].name
            , rows: [ range(0; ($D.fields|length)) as $j
                    | {descr: $D.fields[$j].name, value: $V.values[$i][$j].v} ] } ]
    | map(select(.name=="siena01"))
    | { P_now_W: (map(.rows[]|select(.descr=="ActivePower")|.value)|add),
        E_res_kWh: (map(.rows[]|select(.descr=="FwdActEnergyRes")|.value)|add) }' >> "$OUTPUT_IDLE" 
    sleep 1
done

# TODO:
# 2. capture end energy