#!/bin/bash
# Test aspect ratio support in ALICE
# Tests various aspect ratios including non-divisible-by-8 dimensions

set -e

# Configuration
SERVER="${ALICE_SERVER:-http://localhost:8080}"
MODEL="${ALICE_MODEL:-sd/stable-diffusion-v1-5}"
PROMPT="a cute cat sitting on a windowsill"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing ALICE Aspect Ratio Support"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Server: $SERVER"
echo "Model: $MODEL"
echo ""

# Test cases: [width, height, description]
declare -a tests=(
    "512:512:Square (already divisible by 8)"
    "1024:768:Landscape 4:3 (both divisible by 8)"
    "768:1024:Portrait 3:4 (both divisible by 8)"
    "1920:1080:Full HD 16:9 (both divisible by 8)"
    "640:480:VGA 4:3 (both divisible by 8)"
    "1000:750:Non-standard (should round to 1000x752)"
    "511:511:Odd dimensions (should round to 512x512)"
    "1023:767:Near-standard (should round to 1024x768)"
    "1366:768:Laptop screen (should round to 1368x768)"
)

success_count=0
fail_count=0

for test in "${tests[@]}"; do
    IFS=':' read -r width height description <<< "$test"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Test: $description"
    echo "Requesting: ${width}x${height}"
    echo ""
    
    # Make API request
    response=$(curl -s -X POST "$SERVER/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
            \"samConfig\": {
                \"width\": $width,
                \"height\": $height,
                \"steps\": 5
            }
        }" 2>&1)
    
    # Check if request was successful
    if echo "$response" | grep -q "image_urls"; then
        echo "✅ SUCCESS"
        
        # Extract image URL
        image_url=$(echo "$response" | grep -o '"image_urls":\["[^"]*"' | grep -o 'http[^"]*')
        echo "Image URL: $image_url"
        
        # Extract actual dimensions from metadata if present
        if echo "$response" | grep -q '"width"'; then
            actual_width=$(echo "$response" | grep -o '"width":[0-9]*' | grep -o '[0-9]*$')
            actual_height=$(echo "$response" | grep -o '"height":[0-9]*' | grep -o '[0-9]*$')
            echo "Generated: ${actual_width}x${actual_height}"
            
            # Check if dimensions were rounded
            if [ "$width" != "$actual_width" ] || [ "$height" != "$actual_height" ]; then
                echo "📝 Dimensions rounded from ${width}x${height} to ${actual_width}x${actual_height}"
            fi
        fi
        
        ((success_count++))
    else
        echo "❌ FAILED"
        echo "Response: $response"
        ((fail_count++))
    fi
    
    echo ""
    sleep 1  # Rate limiting
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Successful: $success_count"
echo "Failed: $fail_count"
echo ""

if [ $fail_count -eq 0 ]; then
    echo "✅ All aspect ratio tests passed!"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi
