#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)
# =============================================================================
# ALICE User Collaboration Tool
# =============================================================================
# Purpose: Facilitate collaboration checkpoints between AI agents and users
# Usage:   ./scripts/user_collaboration.sh "Your message here"
# =============================================================================

# Colors for formatting
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

MESSAGE="${1:-Press Enter to continue}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BLUE}🤝 COLLABORATION CHECKPOINT${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "$MESSAGE"
echo ""
echo -en "${GREEN}Enter your response: ${NC}"
read -r response

if [ -n "$response" ]; then
    echo ""
    echo -e "${YELLOW}User response:${NC} $response"
fi

echo ""
echo "Continuing..."
exit 0
