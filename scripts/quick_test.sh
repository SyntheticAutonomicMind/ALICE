#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)
# Quick scheduler test - tests first 3 models with all schedulers

API_BASE="http://127.0.0.1:8090"
TEST_PROMPT="a cute cat"
TEST_STEPS=4
TEST_SIZE=512

SCHEDULERS=("ddim" "dpm++_sde_karras" "euler" "euler_a" "pndm")

echo "Quick Scheduler Test"
echo "===================="

# Get first 3 models
MODELS=$(curl -s "$API_BASE/v1/models" | python3 -c "import sys, json; data=json.load(sys.stdin); print(' '.join([m['id'] for m in data.get('data', [])]))" | tr ' ' '\n' | head -3 | tr '\n' ' ')

PASSED=0
FAILED=0

for MODEL_ID in $MODELS; do
    MODEL_NAME=$(echo $MODEL_ID | sed 's|sd/||g')
    echo "Testing: $MODEL_NAME"
    
    for SCHEDULER in "${SCHEDULERS[@]}"; do
        printf "  %-20s ... " "$SCHEDULER"
        
        PAYLOAD=$(cat <<EOF
{
  "model": "$MODEL_ID",
  "messages": [{"role": "user", "content": "$TEST_PROMPT"}],
  "sam_config": {
    "steps": $TEST_STEPS,
    "width": $TEST_SIZE,
    "height": $TEST_SIZE,
    "scheduler": "$SCHEDULER"
  }
}
EOF
)
        
        START=$(date +%s)
        RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$PAYLOAD" \
            --max-time 120 \
            "$API_BASE/v1/chat/completions")
        END=$(date +%s)
        ELAPSED=$((END - START))
        
        HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
        BODY=$(echo "$RESPONSE" | sed '$d')
        
        if [ "$HTTP_CODE" = "200" ]; then
            IMAGE_URL=$(echo "$BODY" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('choices', [{}])[0].get('message', {}).get('image_urls', [''])[0])" 2>/dev/null)
            if [ -n "$IMAGE_URL" ]; then
                IMG_SIZE=$(curl -s "$IMAGE_URL" | wc -c)
                if [ "$IMG_SIZE" -gt 1000 ]; then
                    echo "✓ PASS (${ELAPSED}s, $((IMG_SIZE/1024))KB)"
                    PASSED=$((PASSED + 1))
                else
                    echo "✗ FAIL - image too small"
                    FAILED=$((FAILED + 1))
                fi
            else
                echo "✗ FAIL - no image URL"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "✗ FAIL - HTTP $HTTP_CODE"
            FAILED=$((FAILED + 1))
        fi
    done
    echo ""
done

echo "===================="
echo "Results: $PASSED passed, $FAILED failed"
